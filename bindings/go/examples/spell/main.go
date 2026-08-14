package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	stridealign "github.com/adamdeprince/stride-align/bindings/go"
)

var dictionaryPaths = []string{
	"/usr/share/dict/words",
	"/usr/dict/words",
	"/var/lib/dict/words",
	"/etc/dictionaries-common/words",
}

func loadDictionary() ([]string, error) {
	var file *os.File
	var err error
	for _, path := range dictionaryPaths {
		file, err = os.Open(path)
		if err == nil {
			break
		}
	}
	if file == nil {
		return nil, fmt.Errorf("no system dictionary found")
	}
	defer file.Close()
	seen := make(map[string]struct{})
	var words []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		word := strings.ToLower(strings.TrimSpace(scanner.Text()))
		if word == "" {
			continue
		}
		if _, exists := seen[word]; !exists {
			seen[word] = struct{}{}
			words = append(words, word)
		}
	}
	return words, scanner.Err()
}

func main() {
	words, err := loadDictionary()
	if err != nil {
		log.Fatal(err)
	}
	input := bufio.NewScanner(os.Stdin)
	if !input.Scan() {
		return
	}
	tokens := strings.Fields(strings.ToLower(input.Text()))
	corrected := make([]string, len(tokens))
	for index, token := range tokens {
		match, ok, matchErr := stridealign.Best(
			token, words, stridealign.ScorerNeedlemanWunschNormalized, nil,
		)
		if matchErr != nil {
			log.Fatal(matchErr)
		}
		if !ok {
			log.Fatal("dictionary is empty")
		}
		corrected[index] = match.Target
	}
	fmt.Println(strings.Join(corrected, " "))
}
