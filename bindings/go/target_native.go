//go:build stridealign_native && !stridealign_avx2 && !stridealign_avx512bwvl && !stridealign_power8_vsx && !stridealign_la464_lsx && !stridealign_la464_lasx && !stridealign_la664_lasx

package stridealign

/*
#cgo CXXFLAGS: -DSTRIDE_ALIGN_GO_TARGET_NATIVE=1
#cgo darwin,arm64 CXXFLAGS: -mcpu=native
#cgo !darwin CXXFLAGS: -march=native
#cgo darwin,amd64 CXXFLAGS: -march=native
*/
import "C"
