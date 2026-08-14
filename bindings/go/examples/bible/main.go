package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	stridealign "github.com/adamdeprince/stride-align/bindings/go"
)

const corpusURL = "https://openbible.com/textfiles/kjv.txt"

func cacheCorpus(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}
	response, err := http.Get(corpusURL)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", corpusURL, response.Status)
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(file, response.Body)
	closeErr := file.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func loadCorpus(path string) ([]string, []string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	var references []string
	var verses []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimPrefix(scanner.Text(), "\ufeff")
		reference, verse, ok := strings.Cut(line, "\t")
		if ok {
			references = append(references, reference)
			verses = append(verses, verse)
		}
	}
	return references, verses, scanner.Err()
}

func main() {
	const corpus = "kjv.txt"
	if err := cacheCorpus(corpus); err != nil {
		log.Fatal(err)
	}
	references, verses, err := loadCorpus(corpus)
	if err != nil {
		log.Fatal(err)
	}
	normalized := make([]string, len(verses))
	for index, verse := range verses {
		normalized[index] = strings.ToLower(verse)
	}
	query := strings.Join([]string{
		"The earth will mourn, and the heavens will lament from above.",
		"For I have spoken, I have decided, and I have not regretted.",
		"Neither will I be turned away from it.",
	}, " ")
	scores, err := stridealign.NeedlemanWunschNormalizedScores(
		strings.ToLower(query), normalized, nil,
	)
	if err != nil {
		log.Fatal(err)
	}
	best := 0
	for index := 1; index < len(scores); index++ {
		if scores[index] > scores[best] {
			best = index
		}
	}
	fmt.Printf("%s\t%s\nscore: %.6f\n", references[best], verses[best], scores[best])
}
