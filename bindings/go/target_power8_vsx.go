//go:build (ppc64 || ppc64le) && stridealign_power8_vsx && !stridealign_native && !stridealign_la464_lsx && !stridealign_la464_lasx && !stridealign_la664_lasx

package stridealign

/*
#cgo CXXFLAGS: -DSTRIDE_ALIGN_GO_TARGET_POWER8_VSX=1 -mcpu=power8 -mtune=power8 -mvsx
*/
import "C"
