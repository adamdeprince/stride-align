//go:build arm64 && !stridealign_native && !stridealign_avx2 && !stridealign_avx512bwvl && !stridealign_power8_vsx && !stridealign_la464_lsx && !stridealign_la464_lasx && !stridealign_la664_lasx

package stridealign

/*
#cgo CXXFLAGS: -DSTRIDE_ALIGN_GO_TARGET_NEON=1
#cgo darwin CXXFLAGS: -mcpu=apple-m1
#cgo linux CXXFLAGS: -march=armv8-a+simd -mtune=generic
*/
import "C"
