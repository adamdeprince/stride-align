#pragma once

#include <cstdint>

namespace stride_align {

using Score = std::int64_t;

enum class KernelBits : std::uint8_t {
  bits8 = 8,
  bits16 = 16,
  bits32 = 32,
  bits64 = 64,
};

}  // namespace stride_align
