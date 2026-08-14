package stridealign

import "strings"

const (
	pairLCSLength = iota
	pairLCSSubstringLength
	pairJaccard
	pairDice
	pairCosine
	pairOverlap
	pairRatcliffObershelp
	pairPartialRatio
	pairTokenSortRatio
	pairTokenSetRatio
	pairPartialTokenSortRatio
	pairPartialTokenSetRatio
	pairWRatio
	pairMongeElkan
)

// Processor transforms an input before a token or fuzzy-ratio operation.
type Processor func(string) (string, error)

// Similarity is a scalar similarity function used by MongeElkanWith.
type Similarity func(string, string) (float64, error)

func pairNumber(operation int, query, target string, n int) (float64, error) {
	return nativePairNumber(operation, query, target, n, "", false)
}

func pairInteger(operation int, query, target string) (int, error) {
	value, err := pairNumber(operation, query, target, 0)
	return int(value), err
}

// LCSLength returns the length in Unicode code points of a longest common subsequence.
func LCSLength(query, target string) (int, error) {
	return pairInteger(pairLCSLength, query, target)
}

// LCSSubstringLength returns the longest common contiguous-substring length.
func LCSSubstringLength(query, target string) (int, error) {
	return pairInteger(pairLCSSubstringLength, query, target)
}

// LCSSubstring returns one longest common contiguous substring.
func LCSSubstring(query, target string) (string, error) {
	return nativeLCSSubstring(query, target)
}

func LCSLengths(query string, targets []string) ([]float64, error) {
	return nativePairNumbers(pairLCSLength, query, targets, 0, "", false)
}

func LCSSubstringLengths(query string, targets []string) ([]float64, error) {
	return nativePairNumbers(pairLCSSubstringLength, query, targets, 0, "", false)
}

func LCSSubstrings(query string, targets []string) ([]string, error) {
	return nativeLCSSubstrings(query, targets)
}

// Jaccard returns n-gram Jaccard similarity.
func Jaccard(query, target string, n int) (float64, error) {
	return pairNumber(pairJaccard, query, target, n)
}

// Dice returns n-gram Dice similarity.
func Dice(query, target string, n int) (float64, error) {
	return pairNumber(pairDice, query, target, n)
}

// Cosine returns n-gram cosine similarity.
func Cosine(query, target string, n int) (float64, error) {
	return pairNumber(pairCosine, query, target, n)
}

// Overlap returns n-gram overlap similarity.
func Overlap(query, target string, n int) (float64, error) {
	return pairNumber(pairOverlap, query, target, n)
}

// RatcliffObershelpSimilarity returns Gestalt pattern-matching similarity.
func RatcliffObershelpSimilarity(query, target string) (float64, error) {
	return pairNumber(pairRatcliffObershelp, query, target, 0)
}

func processedPair(query, target string, processor Processor) (string, string, error) {
	if processor == nil {
		return query, target, nil
	}
	left, err := processor(query)
	if err != nil {
		return "", "", err
	}
	right, err := processor(target)
	if err != nil {
		return "", "", err
	}
	return left, right, nil
}

func processedRatio(operation int, query, target string, processor Processor) (float64, error) {
	query, target, err := processedPair(query, target, processor)
	if err != nil {
		return 0, err
	}
	if operation == pairWRatio && query == "" && target == "" {
		return 1, nil
	}
	return pairNumber(operation, query, target, 0)
}

func PartialRatio(query, target string) (float64, error) {
	return processedRatio(pairPartialRatio, query, target, nil)
}

func PartialRatioWithProcessor(query, target string, processor Processor) (float64, error) {
	return processedRatio(pairPartialRatio, query, target, processor)
}

func TokenSortRatio(query, target string) (float64, error) {
	return processedRatio(pairTokenSortRatio, query, target, nil)
}

func TokenSortRatioWithProcessor(query, target string, processor Processor) (float64, error) {
	return processedRatio(pairTokenSortRatio, query, target, processor)
}

func TokenSetRatio(query, target string) (float64, error) {
	return processedRatio(pairTokenSetRatio, query, target, nil)
}

func TokenSetRatioWithProcessor(query, target string, processor Processor) (float64, error) {
	return processedRatio(pairTokenSetRatio, query, target, processor)
}

func PartialTokenSortRatio(query, target string) (float64, error) {
	return processedRatio(pairPartialTokenSortRatio, query, target, nil)
}

func PartialTokenSortRatioWithProcessor(query, target string, processor Processor) (float64, error) {
	return processedRatio(pairPartialTokenSortRatio, query, target, processor)
}

func PartialTokenSetRatio(query, target string) (float64, error) {
	return processedRatio(pairPartialTokenSetRatio, query, target, nil)
}

func PartialTokenSetRatioWithProcessor(query, target string, processor Processor) (float64, error) {
	return processedRatio(pairPartialTokenSetRatio, query, target, processor)
}

func WRatio(query, target string) (float64, error) {
	return processedRatio(pairWRatio, query, target, nil)
}

func WRatioWithProcessor(query, target string, processor Processor) (float64, error) {
	return processedRatio(pairWRatio, query, target, processor)
}

func processedRatios(operation int, query string, targets []string, processor Processor) ([]float64, error) {
	processedQuery := query
	processedTargets := targets
	if processor != nil {
		var err error
		processedQuery, err = processor(query)
		if err != nil {
			return nil, err
		}
		processedTargets = make([]string, len(targets))
		for index, target := range targets {
			processedTargets[index], err = processor(target)
			if err != nil {
				return nil, err
			}
		}
	}
	values, err := nativePairNumbers(operation, processedQuery, processedTargets, 0, "", false)
	if err != nil {
		return nil, err
	}
	if operation == pairWRatio && processedQuery == "" {
		for index, target := range processedTargets {
			if target == "" {
				values[index] = 1
			}
		}
	}
	return values, nil
}

func PartialRatios(query string, targets []string, processor Processor) ([]float64, error) {
	return processedRatios(pairPartialRatio, query, targets, processor)
}

func TokenSortRatios(query string, targets []string, processor Processor) ([]float64, error) {
	return processedRatios(pairTokenSortRatio, query, targets, processor)
}

func TokenSetRatios(query string, targets []string, processor Processor) ([]float64, error) {
	return processedRatios(pairTokenSetRatio, query, targets, processor)
}

func PartialTokenSortRatios(query string, targets []string, processor Processor) ([]float64, error) {
	return processedRatios(pairPartialTokenSortRatio, query, targets, processor)
}

func PartialTokenSetRatios(query string, targets []string, processor Processor) ([]float64, error) {
	return processedRatios(pairPartialTokenSetRatio, query, targets, processor)
}

func WRatios(query string, targets []string, processor Processor) ([]float64, error) {
	return processedRatios(pairWRatio, query, targets, processor)
}

// MongeElkan computes token-level Monge-Elkan similarity. Inner accepts jaro,
// jaro_winkler, levenshtein_ratio, or indel_ratio.
func MongeElkan(query, target, inner string, symmetric bool) (float64, error) {
	if inner == "" {
		inner = "jaro"
	}
	return nativePairNumber(pairMongeElkan, query, target, 0, inner, symmetric)
}

// MongeElkanWith supports the R binding's custom inner-scorer form.
func MongeElkanWith(query, target string, inner Similarity, processor Processor, symmetric bool) (float64, error) {
	query, target, err := processedPair(query, target, processor)
	if err != nil {
		return 0, err
	}
	if inner == nil {
		return MongeElkan(query, target, "jaro", symmetric)
	}
	left := strings.Fields(query)
	right := strings.Fields(target)
	if len(left) == 0 && len(right) == 0 {
		return 1, nil
	}
	if len(left) == 0 || len(right) == 0 {
		return 0, nil
	}
	direction := func(from, to []string) (float64, error) {
		total := 0.0
		for _, token := range from {
			best := 0.0
			for _, candidate := range to {
				value, scoreErr := inner(token, candidate)
				if scoreErr != nil {
					return 0, scoreErr
				}
				if value > best {
					best = value
				}
			}
			total += best
		}
		return total / float64(len(from)), nil
	}
	forward, err := direction(left, right)
	if err != nil || !symmetric {
		return forward, err
	}
	reverse, err := direction(right, left)
	return (forward + reverse) / 2, err
}

func pairSimilarities(query string, targets []string, operation, n int) ([]float64, error) {
	return nativePairNumbers(operation, query, targets, n, "", false)
}

func JaccardSimilarities(query string, targets []string, n int) ([]float64, error) {
	return pairSimilarities(query, targets, pairJaccard, n)
}

func DiceSimilarities(query string, targets []string, n int) ([]float64, error) {
	return pairSimilarities(query, targets, pairDice, n)
}

func CosineSimilarities(query string, targets []string, n int) ([]float64, error) {
	return pairSimilarities(query, targets, pairCosine, n)
}

func OverlapSimilarities(query string, targets []string, n int) ([]float64, error) {
	return pairSimilarities(query, targets, pairOverlap, n)
}

func RatcliffObershelpSimilarities(query string, targets []string) ([]float64, error) {
	return pairSimilarities(query, targets, pairRatcliffObershelp, 0)
}
