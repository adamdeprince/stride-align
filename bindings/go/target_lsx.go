//go:build loong64 && stridealign_la464_lsx && !stridealign_native && !stridealign_la464_lasx && !stridealign_la664_lasx

package stridealign

/*
#cgo CXXFLAGS: -DSTRIDE_ALIGN_GO_TARGET_LSX=1 -DSTRIDE_ALIGN_GO_TARGET_LA464=1 -march=la464 -mtune=la464 -mlsx -mno-lasx
*/
import "C"
