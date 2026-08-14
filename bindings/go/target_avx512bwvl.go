//go:build amd64 && stridealign_avx512bwvl && !stridealign_native && !stridealign_avx2 && !stridealign_power8_vsx && !stridealign_la464_lsx && !stridealign_la464_lasx && !stridealign_la664_lasx

package stridealign

/*
#cgo CXXFLAGS: -DSTRIDE_ALIGN_GO_TARGET_AVX512BWVL=1 -march=x86-64-v4 -mtune=generic
*/
import "C"
