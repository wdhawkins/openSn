// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/mesh/cell/cell.h"

namespace opensn
{

/// Stores references to global cells to enable an iterator
class LocalCellHandler
{
  friend class MeshContinuum;

public:
  std::vector<std::unique_ptr<Cell>>& native_cells;

private:
  /// Constructor.
  explicit LocalCellHandler(std::vector<std::unique_ptr<Cell>>& native_cells)
    : native_cells(native_cells)
  {
  }

public:
  /// Returns a reference to a local cell, given a local cell index.
  Cell& operator[](uint64_t cell_local_index);

  /// Returns a const reference to a local cell, given a local cell index.
  const Cell& operator[](uint64_t cell_local_index) const;

  /// Internal iterator class.
  class iterator
  {
  public:
    LocalCellHandler& ref_block;
    size_t ref_element;

    iterator(LocalCellHandler& block, size_t i) : ref_block(block), ref_element(i) {}

    iterator operator++()
    {
      iterator i = *this;
      ref_element++;
      return i;
    }
    iterator operator++(int)
    {
      ref_element++;
      return *this;
    }

    Cell& operator*() { return *(ref_block.native_cells[ref_element]); }
    bool operator==(const iterator& rhs) const { return ref_element == rhs.ref_element; }
    bool operator!=(const iterator& rhs) const { return ref_element != rhs.ref_element; }
  };

  /// Internal const iterator class.
  class const_iterator
  {
  public:
    const LocalCellHandler& ref_block;
    size_t ref_element;

    const_iterator(const LocalCellHandler& block, size_t i) : ref_block(block), ref_element(i) {}

    const_iterator operator++()
    {
      const_iterator i = *this;
      ref_element++;
      return i;
    }
    const_iterator operator++(int)
    {
      ref_element++;
      return *this;
    }

    const Cell& operator*() { return *(ref_block.native_cells[ref_element]); }
    bool operator==(const const_iterator& rhs) const { return ref_element == rhs.ref_element; }
    bool operator!=(const const_iterator& rhs) const { return ref_element != rhs.ref_element; }
  };

  iterator begin() { return {*this, 0}; }

  iterator end() { return {*this, native_cells.size()}; }

  const_iterator begin() const { return {*this, 0}; }

  const_iterator end() const { return {*this, native_cells.size()}; }

  size_t size() const { return native_cells.size(); }
};

} // namespace opensn
