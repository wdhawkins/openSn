# Read dependencies.json and define, for each dependency NAME (upper case, with
# '-' replaced by '_'):
#
#   OPENSN_<NAME>_MIN_VERSION         when "minimum" is not null
#   OPENSN_<NAME>_BOOTSTRAP_VERSION   when a "bootstrap" block is present
#   OPENSN_<NAME>_BOOTSTRAP_URL
#   OPENSN_<NAME>_BOOTSTRAP_SHA256

if(NOT DEFINED OPENSN_DEPENDENCY_MANIFEST)
  set(OPENSN_DEPENDENCY_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/../dependencies.json")
endif()

file(READ "${OPENSN_DEPENDENCY_MANIFEST}" _opensn_json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${OPENSN_DEPENDENCY_MANIFEST}")

string(JSON _opensn_count ERROR_VARIABLE _opensn_error LENGTH "${_opensn_json}" dependencies)
if(_opensn_error)
  message(FATAL_ERROR "Invalid ${OPENSN_DEPENDENCY_MANIFEST}: ${_opensn_error}")
endif()

math(EXPR _opensn_last "${_opensn_count} - 1")
foreach(_opensn_index RANGE ${_opensn_last})
  string(JSON _opensn_name MEMBER "${_opensn_json}" dependencies ${_opensn_index})
  string(TOUPPER "${_opensn_name}" _opensn_var)
  string(REPLACE "-" "_" _opensn_var "${_opensn_var}")

  string(JSON _opensn_type ERROR_VARIABLE _opensn_error
    TYPE "${_opensn_json}" dependencies "${_opensn_name}" minimum)
  if(_opensn_error)
    message(FATAL_ERROR "Dependency manifest entry ${_opensn_name} has no minimum field.")
  endif()
  if(NOT _opensn_type STREQUAL "NULL")
    string(JSON OPENSN_${_opensn_var}_MIN_VERSION
      GET "${_opensn_json}" dependencies "${_opensn_name}" minimum)
  endif()

  string(JSON _opensn_type ERROR_VARIABLE _opensn_error
    TYPE "${_opensn_json}" dependencies "${_opensn_name}" bootstrap)
  if(NOT _opensn_error)
    foreach(_opensn_field IN ITEMS version url sha256)
      string(JSON _opensn_value ERROR_VARIABLE _opensn_error
        GET "${_opensn_json}" dependencies "${_opensn_name}" bootstrap ${_opensn_field})
      if(_opensn_error OR _opensn_value STREQUAL "")
        message(FATAL_ERROR
          "Dependency manifest entry ${_opensn_name}.bootstrap has no ${_opensn_field}.")
      endif()
      string(TOUPPER "${_opensn_field}" _opensn_field_var)
      set(OPENSN_${_opensn_var}_BOOTSTRAP_${_opensn_field_var} "${_opensn_value}")
    endforeach()
    string(LENGTH "${OPENSN_${_opensn_var}_BOOTSTRAP_SHA256}" _opensn_length)
    if(NOT OPENSN_${_opensn_var}_BOOTSTRAP_SHA256 MATCHES "^[0-9a-fA-F]+$"
       OR NOT _opensn_length EQUAL 64)
      message(FATAL_ERROR
        "Dependency manifest entry ${_opensn_name}.bootstrap.sha256 is not a SHA-256 digest.")
    endif()
  endif()
endforeach()

unset(_opensn_json)
unset(_opensn_count)
unset(_opensn_error)
unset(_opensn_last)
unset(_opensn_index)
unset(_opensn_name)
unset(_opensn_var)
unset(_opensn_type)
unset(_opensn_field)
unset(_opensn_field_var)
unset(_opensn_value)
unset(_opensn_length)
