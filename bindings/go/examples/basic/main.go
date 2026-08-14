package main

import (
	"fmt"
	"log"

	stridealign "github.com/adamdeprince/stride-align/bindings/go"
)

func main() {
	score, err := stridealign.LevenshteinNormalizedScore("naïve café", "naive cafe")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("similarity: %.6f\n", score)
}
