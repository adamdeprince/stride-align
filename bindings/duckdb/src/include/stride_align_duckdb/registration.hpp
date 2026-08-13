#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb::stride_align_extension {

void RegisterBatchFunctions(ExtensionLoader& loader);
void RegisterAlgorithmFunctions(ExtensionLoader& loader);
void RegisterMatrixFunctions(ExtensionLoader& loader);

}  // namespace duckdb::stride_align_extension
