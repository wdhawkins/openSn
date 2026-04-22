// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/dense_matrix.h"
#include "framework/data_types/vector.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/math/spatial_discretization/finite_element/unit_cell_matrices.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/csda_sweep_helper.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aah_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_structs.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_view.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aah_sweep_chunk.h"
#include "framework/utils/error.h"
#include <algorithm>

namespace opensn
{

struct AAHSweepData
{
  const std::shared_ptr<MeshContinuum>& grid;
  const SpatialDiscretization& discretization;
  const std::vector<UnitCellMatrices>& unit_cell_matrices;
  std::vector<CellLBSView>& cell_transport_views;
  const std::vector<double>& source_moments;
  const LBSGroupset& groupset;
  const BlockID2XSMap& xs;
  const unsigned int num_moments;
  const unsigned int max_num_cell_dofs;
  const unsigned int min_num_cell_dofs;
  const bool save_angular_flux;
  const size_t groupset_angle_group_stride;
  const size_t groupset_group_stride;
  std::vector<double>& destination_phi;
  std::vector<double>& destination_phi_e;
  std::vector<double>& destination_psi;
  bool surface_source_active;
  bool include_rhs_time_term;
  bool csda_enabled;
  DiscreteOrdinatesProblem& problem;
  const std::vector<double>* psi_old; // nullptr for steady sweeps
  unsigned int group_block_size;      // used by fixed-N/AVX path
};

inline size_t
ComputeGroupBlockSize(size_t gs_size)
{
  if (gs_size <= simd_width)
    return gs_size;

  size_t target = 0;
  if (gs_size >= 16 * simd_width)
    target = 4 * simd_width;
  else if (gs_size >= 4 * simd_width)
    target = 2 * simd_width;
  else
    target = 1 * simd_width;

  target = std::min(target, gs_size);
  if (target >= simd_width)
    target = (target / simd_width) * simd_width;
  return target;
}

/// Generic sweep kernel (scalar), parameterized by time dependence.
template <bool time_dependent>
inline void
AAH_Sweep_Generic(AAHSweepData& data, AngleSet& angle_set)
{
  const auto& groupset = data.groupset;
  const auto gs_size = groupset.GetNumGroups();
  const auto gs_gi = groupset.first_group;

  int deploc_face_counter = -1;
  int preloc_face_counter = -1;

  auto& fluds = dynamic_cast<AAH_FLUDS&>(angle_set.GetFLUDS());
  const auto& m2d_op = groupset.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset.quadrature->GetDiscreteToMomentOperator();

  DenseMatrix<double> Amat(data.max_num_cell_dofs, data.max_num_cell_dofs);
  DenseMatrix<double> Atemp(data.max_num_cell_dofs, data.max_num_cell_dofs);
  DenseMatrix<double> Aext(data.max_num_cell_dofs + 1, data.max_num_cell_dofs + 1);
  std::vector<Vector<double>> b(gs_size, Vector<double>(data.max_num_cell_dofs, 0.0));
  Vector<double> b_ext(data.max_num_cell_dofs + 1, 0.0);
  std::vector<double> source(data.max_num_cell_dofs);

  const auto& spds = angle_set.GetSPDS();
  const auto& spls = spds.GetLocalSubgrid();
  const size_t num_spls = spls.size();

  for (size_t spls_index = 0; spls_index < num_spls; ++spls_index)
  {
    const auto cell_local_id = spls[spls_index];
    auto& cell = data.grid->local_cells[cell_local_id];
    auto& cell_transport_view = data.cell_transport_views[cell_local_id];
    const auto& cell_mapping = data.discretization.GetCellMapping(cell);
    const size_t cell_num_faces = cell.faces.size();
    const size_t cell_num_nodes = cell_mapping.GetNumNodes();

    const auto& face_orientations = spds.GetCellFaceOrientations()[cell_local_id];
    std::vector<double> face_mu_values(cell_num_faces);

    const auto& xs = data.xs.at(cell.block_id);
    const auto& sigma_t = xs->GetSigmaTotal();

    const auto csda_enabled = data.csda_enabled;
    const auto csda_data = csda_enabled ? MakeCSDAMaterialData(*xs) : CSDAMaterialData{};

    std::vector<double> tau_gsg;
    if constexpr (time_dependent)
    {
      const auto& inv_velg = data.xs.at(cell.block_id)->GetInverseVelocity();
      const double theta = data.problem.GetTheta();
      const double inv_theta = 1.0 / theta;
      const double dt = data.problem.GetTimeStep();
      const double inv_dt = 1.0 / dt;
      tau_gsg.assign(gs_size, 0.0);
      for (size_t gsg = 0; gsg < gs_size; ++gsg)
        tau_gsg[gsg] = inv_velg[gs_gi + gsg] * inv_theta * inv_dt;
    }

    const auto& G = data.unit_cell_matrices[cell_local_id].intV_shapeI_gradshapeJ;
    const auto& M = data.unit_cell_matrices[cell_local_id].intV_shapeI_shapeJ;
    const auto& v = data.unit_cell_matrices[cell_local_id].intV_shapeI;
    const auto& M_surf = data.unit_cell_matrices[cell_local_id].intS_shapeI_shapeJ;
    const auto& IntF_shapeI = data.unit_cell_matrices[cell_local_id].intS_shapeI;

    double cell_volume = 0.0;
    for (size_t i = 0; i < cell_num_nodes; ++i)
      cell_volume += v(i);

    const int ni_deploc_face_counter = deploc_face_counter;
    const int ni_preloc_face_counter = preloc_face_counter;
    const std::vector<std::uint32_t>& as_angle_indices = angle_set.GetAngleIndices();

    for (size_t as_ss_idx = 0; as_ss_idx < as_angle_indices.size(); ++as_ss_idx)
    {
      const auto direction_num = as_angle_indices[as_ss_idx];
      const auto omega = groupset.quadrature->omegas[direction_num];
      const auto wt = groupset.quadrature->weights[direction_num];

      deploc_face_counter = ni_deploc_face_counter;
      preloc_face_counter = ni_preloc_face_counter;

      // Cell-wise coefficient of the within-group energy basis
      // chi_g(E) = 2 * (E - E_g) / DeltaE_g.
      std::vector<double> psiE_gsg;
      if (csda_enabled)
        psiE_gsg.assign(gs_size, 0.0);

      for (size_t gsg = 0; gsg < gs_size; ++gsg)
        for (size_t i = 0; i < cell_num_nodes; ++i)
          b[gsg](i) = 0.0;

      for (size_t i = 0; i < cell_num_nodes; ++i)
        for (size_t j = 0; j < cell_num_nodes; ++j)
          Amat(i, j) = omega.Dot(G(i, j));

      for (size_t f = 0; f < cell_num_faces; ++f)
        face_mu_values[f] = omega.Dot(cell.faces[f].normal);

      int in_face_counter = -1;
      for (size_t f = 0; f < cell_num_faces; ++f)
      {
        if (face_orientations[f] != FaceOrientation::INCOMING)
          continue;

        auto& cell_face = cell.faces[f];
        const bool is_local_face = cell_transport_view.IsFaceLocal(f);
        const bool is_boundary_face = not cell_face.has_neighbor;

        if (is_local_face)
          ++in_face_counter;
        else if (not is_boundary_face)
          ++preloc_face_counter;

        const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
        for (size_t fi = 0; fi < num_face_nodes; ++fi)
        {
          const int i = cell_mapping.MapFaceNode(f, fi);
          for (size_t fj = 0; fj < num_face_nodes; ++fj)
          {
            const int j = cell_mapping.MapFaceNode(f, fj);
            const double mu_Nij = -face_mu_values[f] * M_surf[f](i, j);
            Amat(i, j) += mu_Nij;

            const double* psi = nullptr;
            if (is_local_face)
              psi = fluds.UpwindPsi(spls_index, in_face_counter, fj, 0, as_ss_idx);
            else if (not is_boundary_face)
              psi = fluds.NLUpwindPsi(preloc_face_counter, fj, 0, as_ss_idx);
            else
              psi = angle_set.PsiBoundary(cell_face.neighbor_id,
                                          direction_num,
                                          cell_local_id,
                                          f,
                                          fj,
                                          gs_gi,
                                          data.surface_source_active);

            if (not psi)
              continue;

            for (size_t gsg = 0; gsg < gs_size; ++gsg)
              b[gsg](i) += psi[gsg] * mu_Nij;
          }
        }
      }

      const double* psi_old =
        (time_dependent and data.psi_old)
          ? &(*data.psi_old)[data.discretization.MapDOFLocal(cell, 0, groupset.psi_uk_man_, 0, 0)]
          : nullptr;
      const double* m2d_row = m2d_op.data() + direction_num * static_cast<size_t>(data.num_moments);
      const double* d2m_row = d2m_op.data() + direction_num * static_cast<size_t>(data.num_moments);

      for (size_t gsg = 0; gsg < gs_size; ++gsg)
      {
        double sigma_tg = sigma_t[gs_gi + gsg];
        if constexpr (time_dependent)
          sigma_tg += tau_gsg[gsg];

        const bool csda_active = IsCSDAActive(csda_data, gs_gi + gsg);

        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          double temp_src = 0.0;
          for (unsigned int m = 0; m < data.num_moments; ++m)
          {
            const size_t ir = cell_transport_view.MapDOF(i, m, gs_gi + gsg);
            temp_src += m2d_row[m] * data.source_moments[ir];
          }

          if constexpr (time_dependent)
          {
            const size_t imap =
              i * data.groupset_angle_group_stride + direction_num * data.groupset_group_stride;
            if (data.include_rhs_time_term and psi_old)
              temp_src += tau_gsg[gsg] * psi_old[imap + gsg];
          }

          source[i] = temp_src;
        }

        if (csda_active)
        {
          const CSDALocalSolveContext csda_ctx{gsg,
                                               gs_gi,
                                               sigma_tg,
                                               cell_volume,
                                               &Amat,
                                               &M,
                                               &v,
                                               &cell,
                                               &cell_mapping,
                                               &face_orientations,
                                               &face_mu_values,
                                               &IntF_shapeI,
                                               &cell_transport_view,
                                               &angle_set,
                                               ni_preloc_face_counter,
                                               &csda_data,
                                               &Aext,
                                               &b_ext,
                                               &psiE_gsg};
          SolveCSDALocalSystem(
            csda_ctx,
            [&source](size_t j) { return source[j]; },
            [&b](size_t gg, size_t j) { return b[gg](j); },
            [&b](size_t gg, size_t i, double value) { b[gg](i) = value; },
            [&fluds, &angle_set, &cell, cell_local_id, direction_num, spls_index, as_ss_idx](
              bool is_local,
              bool is_reflecting,
              size_t face_num,
              int in_face_counter_slope,
              int preloc_face_counter_slope,
              size_t gg)
            {
              const double* psiE = nullptr;
              if (is_local)
                psiE = fluds.UpwindPsiE(spls_index, in_face_counter_slope, 0, 0, as_ss_idx);
              else if (is_reflecting)
                psiE = angle_set.PsiBoundaryE(
                  cell.faces[face_num].neighbor_id, direction_num, cell_local_id, face_num, 0, 0);
              else
                psiE = fluds.NLUpwindPsiE(preloc_face_counter_slope, 0, 0, as_ss_idx);
              return psiE ? psiE[gg] : 0.0;
            });
        }
        else
        {
          for (size_t i = 0; i < cell_num_nodes; ++i)
          {
            double temp = 0.0;
            for (size_t j = 0; j < cell_num_nodes; ++j)
            {
              const double Mij = M(i, j);
              Atemp(i, j) = Amat(i, j) + Mij * sigma_tg;
              temp += Mij * source[j];
            }
            b[gsg](i) += temp;
          }

          GaussElimination(Atemp, b[gsg], cell_num_nodes);
        }
      }

      for (unsigned int m = 0; m < data.num_moments; ++m)
      {
        const auto wn_d2m = d2m_row[m];
        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          const size_t ir = cell_transport_view.MapDOF(i, m, gs_gi);
          for (size_t gsg = 0; gsg < gs_size; ++gsg)
            data.destination_phi[ir + gsg] += wn_d2m * b[gsg](i);
        }
      }

      if (csda_enabled)
      {
        const double wn_d2m = d2m_row[0];
        const size_t cell_g_map = cell_local_id * static_cast<size_t>(data.problem.GetNumGroups()) + gs_gi;
        for (size_t gsg = 0; gsg < gs_size; ++gsg)
          data.destination_phi_e[cell_g_map + gsg] += wn_d2m * psiE_gsg[gsg];
      }

      if (data.save_angular_flux)
      {
        double* psi_new =
          &data
             .destination_psi[data.discretization.MapDOFLocal(cell, 0, groupset.psi_uk_man_, 0, 0)];
        double theta = 1.0;
        double inv_theta = 1.0;
        if constexpr (time_dependent)
        {
          theta = data.problem.GetTheta();
          inv_theta = 1.0 / theta;
        }

        for (size_t i = 0; i < cell_num_nodes; ++i)
        {
          const size_t imap =
            i * data.groupset_angle_group_stride + direction_num * data.groupset_group_stride;
          for (size_t gsg = 0; gsg < gs_size; ++gsg)
          {
            const double psi_sol = b[gsg](i);
            if constexpr (time_dependent)
            {
              const double psi_old_val = psi_old ? psi_old[imap + gsg] : 0.0;
              psi_new[imap + gsg] = inv_theta * (psi_sol + (theta - 1.0) * psi_old_val);
            }
            else
            {
              psi_new[imap + gsg] = psi_sol;
            }
          }
        }
      }

      int out_face_counter = -1;
      for (size_t f = 0; f < cell_num_faces; ++f)
      {
        if (face_orientations[f] != FaceOrientation::OUTGOING)
          continue;

        ++out_face_counter;
        const auto& face = cell.faces[f];
        const bool is_local_face = cell_transport_view.IsFaceLocal(f);
        const bool is_boundary_face = not face.has_neighbor;
        const bool is_reflecting_boundary_face =
          (is_boundary_face and angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting());
        const auto& IntF_shapeI = data.unit_cell_matrices[cell_local_id].intS_shapeI[f];

        if (not is_boundary_face and not is_local_face)
          ++deploc_face_counter;

        const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
        for (size_t fi = 0; fi < num_face_nodes; ++fi)
        {
          const int i = cell_mapping.MapFaceNode(f, fi);

          if (is_boundary_face)
          {
            for (size_t gsg = 0; gsg < gs_size; ++gsg)
              cell_transport_view.AddOutflow(
                f, gs_gi + gsg, wt * face_mu_values[f] * b[gsg](i) * IntF_shapeI(i));
          }

          double* psi = nullptr;
          if (is_local_face)
            psi = fluds.OutgoingPsi(spls_index, out_face_counter, fi, as_ss_idx);
          else if (not is_boundary_face)
            psi = fluds.NLOutgoingPsi(deploc_face_counter, fi, as_ss_idx);
          else if (is_reflecting_boundary_face)
            psi = angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id, f, fi);
          else
            continue;

          if (not is_boundary_face or is_reflecting_boundary_face)
          {
            for (size_t gsg = 0; gsg < gs_size; ++gsg)
              psi[gsg] = b[gsg](i);
          }

          if (csda_enabled)
            StoreOutgoingCSDASlope(
              (not is_boundary_face or is_reflecting_boundary_face),
              psiE_gsg,
              [&]() -> double*
              {
                if (is_local_face)
                  return fluds.OutgoingPsiE(spls_index, out_face_counter, fi, as_ss_idx);
                if (is_reflecting_boundary_face)
                  return angle_set.PsiReflectedE(
                    face.neighbor_id, direction_num, cell_local_id, f, fi);
                if (not is_boundary_face)
                  return fluds.NLOutgoingPsiE(deploc_face_counter, fi, as_ss_idx);
                return nullptr;
              });
        }
      }
    }
  }
}

// Fixed-N kernel (scalar/AVX) is specialized in aah_avx_sweep_chunk.cc.
template <unsigned int NumNodes, bool time_dependent>
void AAH_Sweep_FixedN(AAHSweepData& data, AngleSet& angle_set);

} // namespace opensn
