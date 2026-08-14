//go:build amd64 && !stridealign_native && !stridealign_avx2 && !stridealign_avx512bwvl && !stridealign_power8_vsx && !stridealign_la464_lsx && !stridealign_la464_lasx && !stridealign_la664_lasx

package stridealign

/*
#cgo CXXFLAGS: -DSTRIDE_ALIGN_GO_TARGET_GENERIC=1 -march=x86-64 -mtune=generic -msse2 -mno-avx -mno-avx2 -mno-avx512f
*/
import "C"
