// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/vector3.h"
#include <map>
#include <unordered_map>

namespace opensn
{

/// Manages a vertex map with custom calls.
class VertexHandler
{
  using GlobalIDMap = std::map<uint64_t, Vector3>;

public:
  // Iterators
  GlobalIDMap::iterator begin() { return m_global_id_vertex_map_.begin(); }
  GlobalIDMap::iterator end() { return m_global_id_vertex_map_.end(); }

  GlobalIDMap::const_iterator begin() const { return m_global_id_vertex_map_.begin(); }
  GlobalIDMap::const_iterator end() const { return m_global_id_vertex_map_.end(); }

  // Accessors — O(1) via hash map; std::map node addresses are stable after insertion
  Vector3& operator[](const uint64_t global_id) { return *m_fast_lookup_.at(global_id); }

  const Vector3& operator[](const uint64_t global_id) const
  {
    return *m_fast_lookup_.at(global_id);
  }

  // Utilities
  void Insert(const uint64_t global_id, const Vector3& vec)
  {
    auto [it, inserted] = m_global_id_vertex_map_.emplace(global_id, vec);
    if (inserted)
      m_fast_lookup_.emplace(global_id, &it->second);
  }

  size_t GetNumLocallyStored() const { return m_global_id_vertex_map_.size(); }

  void Clear()
  {
    m_global_id_vertex_map_.clear();
    m_fast_lookup_.clear();
  }

private:
  std::map<uint64_t, Vector3> m_global_id_vertex_map_;
  // Parallel O(1) lookup; pointers into m_global_id_vertex_map_ are stable (map never rebalances)
  std::unordered_map<uint64_t, Vector3*> m_fast_lookup_;
};

} // namespace opensn
