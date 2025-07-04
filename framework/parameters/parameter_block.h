// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/data_types/varying.h"
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <map>

namespace opensn
{

enum class ParameterBlockType
{
  INVALID_VALUE = 0,
  BOOLEAN = 1,
  FLOAT = 3,
  STRING = 4,
  INTEGER = 5,
  USER_DATA = 6,
  ARRAY = 98,
  BLOCK = 99
};

std::string ParameterBlockTypeName(ParameterBlockType type);

class ParameterBlock;

/**
 * A ParameterBlock is a conceptually simple data structure that supports a hierarchy of primitive
 * parameters. There really are just 4 member variables on a ParameterBlock object, they are 1) the
 * type (as an enum), 2) the name of the block, 3) a pointer to a value (which can only be a
 * primitive type), and 4) a vector of child parameters.
 *
 * If a ParameterBlock has a primitive type, i.e., BOOLEAN, FLOAT, STRING, or INTEGER, then the
 * value_ptr will contain a pointer to the value of a primitive type. Otherwise, for types ARRAY and
 * BLOCK, the ParameterBlock will not have a value_ptr and instead the vector member will contain
 * sub-parameters.
 */
class ParameterBlock
{
private:
  ParameterBlockType type_ = ParameterBlockType::BLOCK;
  std::string name_;
  std::shared_ptr<Varying> value_ptr_ = nullptr;
  std::vector<ParameterBlock> parameters_;
  std::string error_origin_scope_ = "Unknown Scope";

public:
  /// Sets the name of the block.
  void SetBlockName(const std::string& name);

  // Helpers
  template <typename T>
  struct IsBool
  {
    static constexpr bool value = std::is_same_v<T, bool>;
  };
  template <typename T>
  struct IsFloat
  {
    static constexpr bool value = std::is_floating_point_v<T>;
  };
  template <typename T>
  struct IsString
  {
    static constexpr bool value = std::is_same_v<T, std::string> or std::is_same_v<T, const char*>;
  };
  template <typename T>
  struct IsInteger
  {
    static constexpr bool value = std::is_integral_v<T> and not std::is_same_v<T, bool>;
  };
  template <typename T>
  struct IsUserData
  {
    static constexpr bool value = (std::is_pointer_v<T> or is_shared_ptr_v<T> or
                                   (std::is_class_v<T> and not std::is_same_v<T, std::string>)) and
                                  (not std::is_same_v<T, const char*>);
  };

  // Constructors
  /// Constructs an empty parameter block with the given name and type BLOCK.
  explicit ParameterBlock(const std::string& name = "");

  /// Derived type constructor
  template <typename T>
  ParameterBlock(const std::string& name, const std::vector<T>& array)
    : type_(ParameterBlockType::ARRAY), name_(name)
  {
    size_t k = 0;
    for (const T& value : array)
      AddParameter(std::to_string(k++), value);
  }

  /// Constructs one of the fundamental types.
  template <typename T>
  explicit ParameterBlock(const std::string& name, T value) : name_(name)
  {
    constexpr bool is_supported = IsBool<T>::value or IsFloat<T>::value or IsString<T>::value or
                                  IsInteger<T>::value or IsUserData<T>::value;

    static_assert(is_supported, "Value type not supported for parameter block");

    if (IsBool<T>::value)
      type_ = ParameterBlockType::BOOLEAN;
    if (IsFloat<T>::value)
      type_ = ParameterBlockType::FLOAT;
    if (IsString<T>::value)
      type_ = ParameterBlockType::STRING;
    if (IsInteger<T>::value)
      type_ = ParameterBlockType::INTEGER;
    if (IsUserData<T>::value)
      type_ = ParameterBlockType::USER_DATA;

    value_ptr_ = std::make_shared<Varying>(value);
  }

  /// Copy constructor
  ParameterBlock(const ParameterBlock& other);

  /// Copy assignment operator
  ParameterBlock& operator=(const ParameterBlock& other);

  /// Move constructor
  ParameterBlock(ParameterBlock&& other) noexcept;

  /// Move assignment operator
  ParameterBlock& operator=(ParameterBlock&& other) noexcept;

  // Accessors
  ParameterBlockType GetType() const;

  /**
   * Returns true if the parameter block comprises a single value of any of the types BOOLEAN,
   * FLOAT, STRING, INTEGER.
   */
  bool IsScalar() const;

  /// Returns a string version of the type.
  std::string GetTypeName() const;
  std::string GetName() const;
  const Varying& GetValue() const;

  /// Returns the number of parameters in a block. This is normally only useful for the ARRAY type.
  size_t GetNumParameters() const;

  /// Returns the sub-parameters of this block.
  const std::vector<ParameterBlock>& GetParameters() const;

  /**
   * Returns whether or not the block has a value. If this block has sub-parameters it should not
   * have a value. This is a good way to check if the block is actually a single value because some
   * Parameter blocks can be passed as empty.
   */
  bool HasValue() const;

  // Mutators

  /// Changes the block type to array, making it accessible via integer keys.
  void ChangeToArray();

  /// Sets a string to be displayed alongside exceptions that give some notion of the origin of the
  /// error.
  void SetErrorOriginScope(const std::string& scope);

  /// Gets a string that allows error messages to print the scope of an error.
  std::string GetErrorOriginScope() const { return error_origin_scope_; }

  // Requirements

  /**
   * Checks that the block is of the given type. If it is not it will throw an exception
   * `std::logic_error`.
   */
  void RequireBlockTypeIs(ParameterBlockType type) const;
  void RequireParameterBlockTypeIs(const std::string& param_name, ParameterBlockType type) const
  {
    GetParam(param_name).RequireBlockTypeIs(type);
  }

  /// Check that the parameter with the given name exists otherwise throws a `std::logic_error`.
  void RequireParameter(const std::string& param_name) const;

  // utilities

  /// Adds a parameter to the sub-parameter list.
  void AddParameter(ParameterBlock block);

  /// Makes a ParameterBlock and adds it to the sub-parameters list.
  template <typename T>
  void AddParameter(const std::string& name, T value)
  {
    AddParameter(ParameterBlock(name, value));
  }

  /// Sorts the sub-parameter list according to name. This is useful for regression testing.
  void SortParameters();

  /// Returns true if a parameter with the specified name is in the list of sub-parameters.
  /// Otherwise, false.
  bool Has(const std::string& param_name) const;

  /// Gets a parameter by name.
  ParameterBlock& GetParam(const std::string& param_name);

  /// Gets a parameter by index.
  ParameterBlock& GetParam(size_t index);

  /// Gets a parameter by name.
  const ParameterBlock& GetParam(const std::string& param_name) const;

  /// Gets a parameter by index.
  const ParameterBlock& GetParam(size_t index) const;

  /// Returns the value of the parameter.
  template <typename T>
  T GetValue() const
  {
    if (value_ptr_ == nullptr)
      throw std::logic_error(error_origin_scope_ + std::string(__PRETTY_FUNCTION__) +
                             ": Value not available for block type " +
                             ParameterBlockTypeName(GetType()));
    try
    {
      return GetValue().GetValue<T>();
    }
    catch (const std::exception& exc)
    {
      throw std::logic_error(error_origin_scope_ + ":" + GetName() + " " + exc.what());
    }
  }

  /// Fetches the parameter with the given name and returns it value.
  template <typename T>
  T GetParamValue(const std::string& param_name) const
  {
    try
    {
      const auto& param = GetParam(param_name);
      return param.GetValue<T>();
    }
    catch (const std::out_of_range& oor)
    {
      throw std::out_of_range(error_origin_scope_ + std::string(__PRETTY_FUNCTION__) +
                              ": Parameter \"" + param_name + "\" not present in block");
    }
  }

  /**
   * Fetches the parameter of type std::shared_ptr<T> with the given name and returns its value.
   *
   * Will perform checking on whether or not the pointed-to-object is null (if \p check = true)
   *
   * The optional second template argument can be used to attempt to cast the object
   * to the derived type and will throw an exception if the cast fails.
   */
  template <typename T, typename Derived = T>
  std::shared_ptr<Derived> GetSharedPtrParam(const std::string& param_name,
                                             const bool check = true) const
  {
    static_assert(std::is_base_of_v<T, Derived>, "T is not a base of derived");

    auto value = this->GetParamValue<std::shared_ptr<T>>(param_name);
    if (!value)
    {
      if (check)
        throw std::logic_error(error_origin_scope_ + std::string(__PRETTY_FUNCTION__) +
                               ": shared_ptr param is null");
      return nullptr;
    }
    if constexpr (!std::is_same_v<T, Derived>)
    {
      if (auto derived_value = std::dynamic_pointer_cast<Derived>(value))
        return derived_value;

      throw std::logic_error(error_origin_scope_ + std::string(__PRETTY_FUNCTION__) +
                             ": Supplied object is not derived from " + typeid(T).name());
    }
    else
    {
      return value;
    }
  }

  /**
   * Converts the parameters of an array-type parameter block to a vector of primitive types and
   * returns it.
   */
  template <typename T>
  std::vector<T> GetVectorValue() const
  {
    if (GetType() != ParameterBlockType::ARRAY)
      throw std::logic_error(error_origin_scope_ + std::string(__PRETTY_FUNCTION__) +
                             ": Invalid type requested for parameter of type " +
                             ParameterBlockTypeName(GetType()));

    std::vector<T> vec;
    if (parameters_.empty())
      return vec;

    // Check the first sub-param is of the right type
    const auto& front_param = parameters_.front();

    // Check that all other parameters are of the required type
    for (const auto& param : parameters_)
      if (param.GetType() != front_param.GetType())
        throw std::logic_error(error_origin_scope_ + " " + std::string(__PRETTY_FUNCTION__) +
                               ": Parameter \"" + name_ +
                               "\", cannot construct vector from block because "
                               "the sub_parameters do not all have the correct type. param->" +
                               ParameterBlockTypeName(param.GetType()) + " vs param0->" +
                               ParameterBlockTypeName(front_param.GetType()));

    const size_t num_params = parameters_.size();
    for (size_t k = 0; k < num_params; ++k)
    {
      const auto& param = GetParam(k);
      vec.push_back(param.GetValue<T>());
    }

    return vec;
  }

  /// Gets a vector of primitive types from an array-type parameter block specified as a parameter
  /// of the current block.
  template <typename T>
  std::vector<T> GetParamVectorValue(const std::string& param_name) const
  {
    const auto& param = GetParam(param_name);
    return param.GetVectorValue<T>();
  }

  // Iterator
  class Iterator
  {
  public:
    ParameterBlock& ref_block;
    size_t ref_id;

    Iterator(ParameterBlock& block, size_t i) : ref_block(block), ref_id(i) {}

    Iterator operator++()
    {
      Iterator i = *this;
      ref_id++;
      return i;
    }
    Iterator operator++(int)
    {
      ref_id++;
      return *this;
    }

    ParameterBlock& operator*() { return ref_block.parameters_[ref_id]; }
    bool operator==(const Iterator& rhs) const { return ref_id == rhs.ref_id; }
    bool operator!=(const Iterator& rhs) const { return ref_id != rhs.ref_id; }
  };

  class ConstIterator
  {
  public:
    const ParameterBlock& ref_block;
    size_t ref_id;

    ConstIterator(const ParameterBlock& block, size_t i) : ref_block(block), ref_id(i) {}

    ConstIterator operator++()
    {
      ConstIterator i = *this;
      ref_id++;
      return i;
    }
    ConstIterator operator++(int)
    {
      ref_id++;
      return *this;
    }

    const ParameterBlock& operator*() { return ref_block.parameters_[ref_id]; }
    bool operator==(const ConstIterator& rhs) const { return ref_id == rhs.ref_id; }
    bool operator!=(const ConstIterator& rhs) const { return ref_id != rhs.ref_id; }
  };

  Iterator begin() { return {*this, 0}; }
  Iterator end() { return {*this, parameters_.size()}; }

  ConstIterator begin() const { return {*this, 0}; }
  ConstIterator end() const { return {*this, parameters_.size()}; }

  /**
   * Given a reference to a string, recursively travels the parameter tree and print values into
   * the reference string.
   */
  void RecursiveDumpToString(std::string& outstr, const std::string& offset = "") const;

  /// Print the block tree structure into a designated string.
  void RecursiveDumpToJSON(std::string& outstr) const;
};

} // namespace opensn
