//go:build !stridealign_prebuilt

// cgo only compiles C++ translation units in the Go package directory.
// Include the shared implementation here so the Go module still consumes the
// same engine rather than maintaining a binding-specific copy.
#include "../../src/cpp/beider_morse_impl.cpp"
