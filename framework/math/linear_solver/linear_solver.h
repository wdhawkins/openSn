// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/linear_solver/linear_solver_context.h"
#include <memory>

namespace opensn
{

struct LinearSolverContext;

class LinearSolver
{
public:
  LinearSolver(std::shared_ptr<LinearSolverContext> context_ptr) : context_ptr_(context_ptr) {}

  virtual ~LinearSolver() {}

  std::shared_ptr<LinearSolverContext>& GetContext() { return context_ptr_; }

  /// Solve the system
  virtual void Setup() {}

  virtual void Solve() = 0;

protected:
  std::shared_ptr<LinearSolverContext> context_ptr_;
};

} // namespace opensn
