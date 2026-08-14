//go:build loong64 && stridealign_la664_lasx && !stridealign_native && !stridealign_la464_lsx && !stridealign_la464_lasx

package stridealign

/*
#cgo CXXFLAGS: -DSTRIDE_ALIGN_GO_TARGET_LASX=1 -DSTRIDE_ALIGN_GO_TARGET_LA664=1 -march=la664 -mtune=la664 -mlsx -mlasx
*/
import "C"
