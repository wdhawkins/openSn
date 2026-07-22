// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/discrete_ordinates_keigen_acceleration.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/cmfd_coarse_mesh.h"
#include <petscksp.h>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace opensn
{

class MultiGroupXS;

class CMFDAcceleration : public DiscreteOrdinatesKEigenAcceleration
{
public:
  explicit CMFDAcceleration(const InputParameters& params);
  ~CMFDAcceleration() override;

  void Initialize() final;
  void PreExecute() final;
  void PrePowerIteration() final;
  double PostPowerIteration() final;
  void PostExecute(bool converged,
                   unsigned int num_power_iterations,
                   std::size_t num_sweeps,
                   double k_eff,
                   double k_eff_change) final;
  bool AllowsPowerIterationConvergence() const final { return last_update_allows_convergence_; }
  std::string GetPowerIterationConvergenceInfo() const final;

  const CMFDCoarseMesh& GetCoarseMesh() const { return coarse_mesh_; }

private:
  struct BalanceResidual
  {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
  };

  struct MLSample
  {
    std::string fingerprint;
    double score = std::numeric_limits<double>::infinity();
    unsigned int aggregation_size = 1;
    unsigned int group_aggregation_size = 1;
    double relaxation = 1.0;
    std::vector<double> features;
  };

  struct MLState
  {
    bool loaded = false;
    unsigned int runs = 0;
    double spatial_weight = 0.0;
    double group_weight = 0.0;
    double relaxation_weight = 0.0;
    std::vector<double> spatial_feature_weights;
    std::vector<double> group_feature_weights;
    std::vector<double> relaxation_feature_weights;
    unsigned int best_aggregation_size = 0;
    unsigned int best_group_aggregation_size = 0;
    double best_relaxation = 0.0;
    double best_score = std::numeric_limits<double>::infinity();
    std::map<std::string, double> baseline_scores;
    std::map<std::string, MLSample> best_samples_by_fingerprint;
  };

  struct MLAction
  {
    unsigned int aggregation_size = 1;
    unsigned int group_aggregation_size = 1;
    double relaxation = 1.0;
  };

  struct CoarseMeshDiagnostics
  {
    std::size_t local_fine_cells = 0;
    std::size_t local_coarse_cells = 0;
    std::size_t local_max_fine_cells_per_coarse_cell = 0;
    std::size_t local_max_faces_per_coarse_cell = 0;
    std::size_t local_undersized_coarse_cells = 0;
    double local_faces_per_coarse_cell_sum = 0.0;
    std::size_t global_fine_cells = 0;
    std::size_t global_coarse_cells = 0;
    std::size_t global_max_fine_cells_per_coarse_cell = 0;
    std::size_t global_max_faces_per_coarse_cell = 0;
    std::size_t global_undersized_coarse_cells = 0;
    double global_faces_per_coarse_cell_sum = 0.0;
  };

  struct FluxUpdateDiagnostics
  {
    double min_scalar_flux = std::numeric_limits<double>::max();
    bool has_nonfinite_flux = false;
    bool has_invalid_k = false;
  };

  void UpdateAutomaticClosure(BalanceResidual operator_residual,
                              BalanceResidual transport_current_residual,
                              double transport_k_eff,
                              double coarse_k_eff,
                              double applied_damping,
                              bool skipped_correction);
  void ProbeAutomaticClosure(double k_eff,
                             const std::vector<double>& coarse_phi,
                             const std::vector<double>& coarse_source_phi);
  void InitializeLinearSystem();
  void ConfigureCoarseLinearSolver(bool direct);
  void ConfigureTransportSolve(unsigned int max_iterations, double residual_tolerance);
  void BuildCoarseFluxCache();
  void BuildFaceCurrentCache();
  void AssembleOperator();
  void AssembleRHS(Vec rhs, double k_eff, const std::vector<double>& coarse_source_phi) const;
  std::vector<double> SolveCoarseSystem(double k_eff,
                                        const std::vector<double>& coarse_source_phi) const;
  BalanceResidual ComputeBalanceResidual(double k_eff,
                                         const std::vector<double>& coarse_phi,
                                         const std::vector<double>& coarse_source_phi) const;
  BalanceResidual
  ComputeTransportCurrentBalanceResidual(double k_eff,
                                         const std::vector<double>& coarse_phi,
                                         const std::vector<double>& coarse_source_phi) const;
  double ComputeCoarseFissionProduction(const std::vector<double>& coarse_phi) const;
  std::size_t LocalUnknownCount() const;
  std::size_t GlobalUnknownCount() const;
  PetscInt MapDOF(uint64_t coarse_cell_global_id, unsigned int coarse_group) const;
  double ComputeOutwardCurrent(const CMFDCoarseCell& coarse_cell,
                               std::size_t face_index,
                               unsigned int coarse_group) const;
  std::pair<double, double> ComputePartialOutwardCurrents(const CMFDCoarseCell& coarse_cell,
                                                          std::size_t face_index,
                                                          unsigned int coarse_group) const;
  unsigned int FineGroupBegin(unsigned int coarse_group) const;
  unsigned int FineGroupEnd(unsigned int coarse_group) const;
  double CondensedRemovalXS(const MultiGroupXS& xs,
                            const std::vector<double>& fine_coarse_phi,
                            const CMFDCoarseCell& coarse_cell,
                            unsigned int coarse_group) const;
  double CondensedDiffusionCoefficient(const MultiGroupXS& xs,
                                       const std::vector<double>& fine_coarse_phi,
                                       const CMFDCoarseCell& coarse_cell,
                                       unsigned int coarse_group) const;
  double CondensedScatterXS(const MultiGroupXS& xs,
                            const std::vector<double>& fine_coarse_phi,
                            const CMFDCoarseCell& coarse_cell,
                            unsigned int row_coarse_group,
                            unsigned int col_coarse_group) const;
  double CondensedProductionXS(const MultiGroupXS& xs,
                               const std::vector<double>& fine_coarse_phi,
                               const CMFDCoarseCell& coarse_cell,
                               unsigned int row_coarse_group,
                               unsigned int col_coarse_group) const;
  FluxUpdateDiagnostics AnalyzeFluxUpdate(const std::vector<double>& phi, double k_eff) const;
  void ConfigureMLAggregation();
  std::vector<double> ExtractMLFeatures() const;
  std::string BuildMLProblemSignature() const;
  std::string ComputeMLProblemFingerprint(const std::string& signature) const;
  double PredictMLValue(const std::vector<double>& weights, double fallback) const;
  std::tuple<double, double, double> PredictMLActionFromSamples() const;
  double MLFeatureDistance(const std::vector<double>& lhs, const std::vector<double>& rhs) const;
  MLState LoadMLState();
  void SaveMLState(const MLState& state) const;
  void UpdateMLFeatureWeights(MLState& state) const;
  void AddMLSample(MLState& state, double score);
  unsigned int RepairAggregationSize(double value) const;
  unsigned int RepairGroupAggregationSize(double value) const;
  double RepairRelaxation(double value) const;
  MLAction RepairMLAction(double aggregation_size,
                          double group_aggregation_size,
                          double relaxation) const;
  bool HasTriedMLAction(const MLAction& action) const;
  MLAction SelectUntriedExplorationAction(const MLAction& base_action,
                                          double raw_aggregation_size,
                                          double raw_group_aggregation_size,
                                          double raw_relaxation);
  double ComputeMLScore(bool converged,
                        unsigned int num_power_iterations,
                        std::size_t num_sweeps,
                        double k_eff_change) const;
  double ApplyFluxCorrectionWithDamping(const std::vector<double>& transport_phi_new,
                                        const std::vector<double>& coarse_phi,
                                        double old_fission_production,
                                        double transport_k_eff,
                                        double& applied_damping,
                                        unsigned int& attempts,
                                        FluxUpdateDiagnostics& diagnostics,
                                        bool& skipped_correction,
                                        std::string& skip_reason);

  const std::string coarse_mesh_type_;
  const std::string current_closure_;
  const bool automatic_closure_;
  unsigned int aggregation_size_;
  unsigned int group_aggregation_size_;
  double relaxation_;
  const unsigned int correction_max_attempts_;
  const double correction_min_damping_;
  const double negative_flux_tolerance_;
  const unsigned int inactive_iterations_;
  const unsigned int update_wgs_max_its_;
  const double update_wgs_abs_tol_;
  const double balance_residual_tolerance_;
  const std::string coarse_solver_policy_;
  const std::size_t coarse_direct_solve_threshold_;
  const unsigned int first_group_ = 0;
  const unsigned int num_groups_;
  unsigned int num_coarse_groups_;
  const bool ml_enabled_;
  const bool ml_record_baseline_;
  const std::string ml_state_file_;
  const double ml_learning_rate_;
  const double ml_explore_;
  const unsigned int ml_max_aggregation_size_;
  const unsigned int ml_max_group_aggregation_size_;
  const double ml_min_relaxation_;
  const double ml_max_relaxation_;
  MLState ml_state_;
  std::vector<MLSample> ml_samples_;
  std::vector<double> ml_features_;
  std::string ml_problem_signature_;
  std::string ml_problem_fingerprint_;
  double ml_raw_aggregation_size_ = 0.0;
  double ml_raw_group_aggregation_size_ = 0.0;
  double ml_raw_relaxation_ = 1.0;
  unsigned int ml_repair_distance_ = 0;
  bool ml_raw_repaired_action_was_tried_ = false;
  bool ml_exploration_replacement_ = false;
  bool ml_repeated_action_ = false;
  std::size_t ml_tried_actions_for_problem_ = 0;
  std::size_t ml_exploration_candidates_ = 0;
  unsigned int ml_skipped_corrections_ = 0;
  unsigned int ml_damped_corrections_ = 0;
  unsigned int outer_iteration_ = 0;
  bool last_update_allows_convergence_ = true;
  bool last_balance_residual_known_ = false;
  bool last_correction_skipped_ = false;
  std::string last_correction_skip_reason_;
  double last_transport_current_balance_relative_l2_ = std::numeric_limits<double>::infinity();
  std::string active_current_closure_;
  bool automatic_closure_probed_ = false;
  unsigned int automatic_closure_probe_iterations_ = 0;
  double current_closure_blend_ = 0.0;
  double last_restrict_old_time_ = 0.0;
  double transport_start_time_ = 0.0;
  double old_fission_production_ = 0.0;
  unsigned int consecutive_skipped_corrections_ = 0;
  bool using_direct_coarse_solver_ = false;
  CMFDCoarseMesh coarse_mesh_;
  std::vector<double> coarse_phi_old_fine_;
  std::vector<double> coarse_phi_new_fine_;
  std::vector<double> coarse_phi_old_;
  std::vector<double> coarse_phi_new_;
  std::map<std::tuple<uint64_t, unsigned int>, double> coarse_phi_cache_;
  std::map<std::tuple<uint64_t, std::size_t, uint64_t, unsigned int>, double> face_current_cache_;

  Mat A_ = nullptr;
  Vec rhs_ = nullptr;
  KSP ksp_ = nullptr;
  mutable int last_ksp_iterations_ = 0;
  mutable bool last_ksp_converged_ = true;

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<CMFDAcceleration> Create(const ParameterBlock& params);
};

} // namespace opensn
