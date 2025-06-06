// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include "stringstream_color.h"

namespace opensn
{

/// Log stream for adding header information to a string stream.
class LogStream : public std::stringstream
{
private:
  std::ostream* log_stream_;
  std::string log_header_;
  const bool dummy_;
  bool use_color_;

public:
  LogStream(std::ostream* output_stream,
            std::string header,
            bool dummy_flag = false,
            bool use_color = false)
    : log_stream_(output_stream),
      log_header_(std::move(header)),
      dummy_(dummy_flag),
      use_color_(use_color)
  {
  }

  LogStream(const LogStream&) = delete;
  LogStream& operator=(const LogStream&) = delete;

  ~LogStream()
  {
    if (dummy_)
      return;

    std::string content = this->str();
    if (content.empty())
      return;

    std::istringstream iss(content);
    std::string line;
    std::string oline;
    std::string reset_str = use_color_ ? StringStreamColor(StringStreamColorCode::RESET) : "";
    while (std::getline(iss, line))
      oline += log_header_ + line + reset_str + "\n";

    if (!oline.empty())
      *log_stream_ << oline << std::flush;
  }
};

struct DummyStream : public std::ostream
{
  struct DummyStreamBuffer : std::streambuf
  {
    virtual int overflow(int c) { return c; };
  } buffer;

  DummyStream() : std::ostream(&buffer) {}
  ~DummyStream() {}
};

} // namespace opensn
