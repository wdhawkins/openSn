// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "lua/modules/linear_bolzmann_solvers/lbs_solver/lbs_common_lua_functions.h"
#include "lua/framework/lua.h"
#include "lua/framework/console/console.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/lbs_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/io/lbs_solver_io.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"

using namespace opensn;

namespace opensnlua
{

RegisterLuaFunctionInNamespace(LBSWriteFluxMoments, lbs, WriteFluxMoments);
RegisterLuaFunctionInNamespace(LBSCreateAndWriteSourceMoments, lbs, CreateAndWriteSourceMoments);
RegisterLuaFunctionInNamespace(LBSReadFluxMomentsAndMakeSourceMoments,
                               lbs,
                               ReadFluxMomentsAndMakeSourceMoments);
RegisterLuaFunctionInNamespace(LBSReadSourceMoments, lbs, ReadSourceMoments);
RegisterLuaFunctionInNamespace(LBSReadFluxMoments, lbs, ReadFluxMoments);

int
LBSWriteFluxMoments(lua_State* L)
{
  const std::string fname = "lbs.WriteFluxMoments";
  LuaCheckArgs<int, std::string>(L, fname);

  const auto solver_handle = LuaArg<int>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);
  LBSSolverIO::WriteFluxMoments(lbs_solver, file_base);

  return LuaReturn(L);
}

int
LBSCreateAndWriteSourceMoments(lua_State* L)
{
  const std::string fname = "lbs.CreateAndWriteSourceMoments";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);

  auto source_moments = lbs_solver.MakeSourceMomentsFromPhi();
  LBSSolverIO::WriteFluxMoments(lbs_solver, file_base, source_moments);

  return LuaReturn(L);
}

int
LBSReadFluxMomentsAndMakeSourceMoments(lua_State* L)
{
  const std::string fname = "lbs.ReadFluxMomentsAndMakeSourceMoments";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);
  bool single_file_flag = LuaArgOptional<bool>(L, 3, false);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);

  LBSSolverIO::ReadFluxMoments(
    lbs_solver, file_base, single_file_flag, lbs_solver.ExtSrcMomentsLocal());

  opensn::log.Log() << "Making source moments from flux file.";
  auto temp_phi = lbs_solver.PhiOldLocal();
  lbs_solver.PhiOldLocal() = lbs_solver.ExtSrcMomentsLocal();
  lbs_solver.ExtSrcMomentsLocal() = lbs_solver.MakeSourceMomentsFromPhi();
  lbs_solver.PhiOldLocal() = temp_phi;

  return LuaReturn(L);
}

int
LBSReadSourceMoments(lua_State* L)
{
  const std::string fname = "lbs.ReadSourceMoments";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);
  auto single_file_flag = LuaArgOptional<bool>(L, 3, false);

  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);

  LBSSolverIO::ReadFluxMoments(
    lbs_solver, file_base, single_file_flag, lbs_solver.ExtSrcMomentsLocal());

  return LuaReturn(L);
}

int
LBSReadFluxMoments(lua_State* L)
{
  const std::string fname = "lbs.ReadFluxMoments";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);
  auto single_file_flag = LuaArgOptional<bool>(L, 3, false);

  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);
  LBSSolverIO::ReadFluxMoments(lbs_solver, file_base, single_file_flag);

  return LuaReturn(L);
}

} // namespace opensnlua
