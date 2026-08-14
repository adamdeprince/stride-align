// Package stridealign exposes stride-align's native C++ string-alignment
// algorithms to Go. Strings are interpreted as UTF-8 and may contain NUL.
package stridealign

import (
	"fmt"
	"strings"
)

// Scorer identifies one of the scorers supported by the generic batch API.
// Its values are stable across the Python, R, Go, DuckDB, PostgreSQL, and Memgraph
// bindings.
type Scorer int

const (
	ScorerLevenshtein Scorer = iota
	ScorerLevenshteinNormalized
	ScorerDamerauLevenshtein
	ScorerDamerauLevenshteinNormalized
	ScorerHamming
	ScorerHammingNormalized
	ScorerJaro
	ScorerJaroWinkler
	ScorerIndel
	ScorerIndelNormalized
	ScorerTrueDamerauLevenshtein
	ScorerTrueDamerauLevenshteinNormalized
	ScorerSmithWaterman
	ScorerSmithWatermanNormalized
	ScorerNeedlemanWunsch
	ScorerNeedlemanWunschNormalized
)

// ScoreOptions controls alignment scores and Jaro-Winkler prefix handling.
// A nil *ScoreOptions uses DefaultScoreOptions.
type ScoreOptions struct {
	MatchScore      int64
	MismatchScore   int64
	GapOpenScore    int64
	GapExtendScore  int64
	PrefixWeight    float64
	PrefixThreshold float64
	PrefixCap       int
}

// DefaultScoreOptions returns the options used by the other stride-align
// language bindings.
func DefaultScoreOptions() ScoreOptions {
	return ScoreOptions{
		MatchScore:      2,
		MismatchScore:   -1,
		GapOpenScore:    -1,
		GapExtendScore:  -1,
		PrefixWeight:    0.1,
		PrefixThreshold: 0.7,
		PrefixCap:       4,
	}
}

// Version returns the lockstep stride-align release version.
func Version() string { return nativeVersion() }

// SIMDLevel reports the C++ target compiled into this Go package.
func SIMDLevel() string { return nativeSIMDLevel() }

// ParseScorer resolves the canonical scorer names and common aliases used by
// the other bindings.
func ParseScorer(name string) (Scorer, error) {
	canonical := strings.ToLower(strings.ReplaceAll(name, "-", "_"))
	canonical = strings.TrimPrefix(canonical, "stride_")
	aliases := map[string]Scorer{
		"0": ScorerLevenshtein, "levenshtein": ScorerLevenshtein,
		"levenshtein_score":            ScorerLevenshtein,
		"1":                            ScorerLevenshteinNormalized,
		"levenshtein_normalized":       ScorerLevenshteinNormalized,
		"levenshtein_normalized_score": ScorerLevenshteinNormalized,
		"2":                            ScorerDamerauLevenshtein, "osa": ScorerDamerauLevenshtein,
		"damerau_levenshtein":                  ScorerDamerauLevenshtein,
		"damerau_levenshtein_score":            ScorerDamerauLevenshtein,
		"3":                                    ScorerDamerauLevenshteinNormalized,
		"osa_normalized":                       ScorerDamerauLevenshteinNormalized,
		"damerau_levenshtein_normalized":       ScorerDamerauLevenshteinNormalized,
		"damerau_levenshtein_normalized_score": ScorerDamerauLevenshteinNormalized,
		"4":                                    ScorerHamming, "hamming": ScorerHamming,
		"hamming_score":            ScorerHamming,
		"5":                        ScorerHammingNormalized,
		"hamming_normalized":       ScorerHammingNormalized,
		"hamming_normalized_score": ScorerHammingNormalized,
		"6":                        ScorerJaro, "jaro": ScorerJaro, "jaro_similarity": ScorerJaro,
		"7": ScorerJaroWinkler, "jaro_winkler": ScorerJaroWinkler,
		"jaro_winkler_similarity": ScorerJaroWinkler,
		"8":                       ScorerIndel, "indel": ScorerIndel, "indel_score": ScorerIndel,
		"9": ScorerIndelNormalized, "indel_normalized": ScorerIndelNormalized,
		"indel_normalized_score":              ScorerIndelNormalized,
		"10":                                  ScorerTrueDamerauLevenshtein,
		"true_damerau_levenshtein":            ScorerTrueDamerauLevenshtein,
		"true_damerau_levenshtein_score":      ScorerTrueDamerauLevenshtein,
		"11":                                  ScorerTrueDamerauLevenshteinNormalized,
		"true_damerau_levenshtein_normalized": ScorerTrueDamerauLevenshteinNormalized,
		"true_damerau_levenshtein_normalized_score": ScorerTrueDamerauLevenshteinNormalized,
		"12": ScorerSmithWaterman, "sw": ScorerSmithWaterman,
		"local": ScorerSmithWaterman, "smith_waterman": ScorerSmithWaterman,
		"smith_waterman_score":            ScorerSmithWaterman,
		"13":                              ScorerSmithWatermanNormalized,
		"smith_waterman_normalized":       ScorerSmithWatermanNormalized,
		"smith_waterman_normalized_score": ScorerSmithWatermanNormalized,
		"14":                              ScorerNeedlemanWunsch, "nw": ScorerNeedlemanWunsch,
		"global": ScorerNeedlemanWunsch, "needleman_wunsch": ScorerNeedlemanWunsch,
		"needleman_wunsch_score":            ScorerNeedlemanWunsch,
		"15":                                ScorerNeedlemanWunschNormalized,
		"needleman_wunsch_normalized":       ScorerNeedlemanWunschNormalized,
		"needleman_wunsch_normalized_score": ScorerNeedlemanWunschNormalized,
	}
	if scorer, ok := aliases[canonical]; ok {
		return scorer, nil
	}
	return 0, fmt.Errorf("stridealign: unknown scorer %q", name)
}

// IsSimilarity reports whether larger values are better for the scorer.
func (scorer Scorer) IsSimilarity() bool {
	return scorer != ScorerLevenshtein &&
		scorer != ScorerDamerauLevenshtein &&
		scorer != ScorerHamming &&
		scorer != ScorerIndel &&
		scorer != ScorerTrueDamerauLevenshtein
}

// IsNormalized reports whether the scorer produces a normalized similarity.
func (scorer Scorer) IsNormalized() bool {
	switch scorer {
	case ScorerLevenshteinNormalized, ScorerDamerauLevenshteinNormalized,
		ScorerHammingNormalized, ScorerJaro, ScorerJaroWinkler,
		ScorerIndelNormalized, ScorerTrueDamerauLevenshteinNormalized,
		ScorerSmithWatermanNormalized, ScorerNeedlemanWunschNormalized:
		return true
	default:
		return false
	}
}

// Score compares two strings with any generic scorer.
func Score(query, target string, scorer Scorer, options *ScoreOptions) (float64, error) {
	return nativeScore(query, target, scorer, options)
}

func integerScore(query, target string, scorer Scorer, options *ScoreOptions) (int64, error) {
	value, err := Score(query, target, scorer, options)
	return int64(value), err
}

// LevenshteinScore returns the Levenshtein edit distance.
func LevenshteinScore(query, target string) (int64, error) {
	return integerScore(query, target, ScorerLevenshtein, nil)
}

// LevenshteinScoreCutoff returns cutoff+1 when the distance exceeds cutoff.
func LevenshteinScoreCutoff(query, target string, cutoff int64) (int64, error) {
	if cutoff < 0 {
		return 0, fmt.Errorf("stridealign: score cutoff must be non-negative")
	}
	value, err := LevenshteinScore(query, target)
	if err == nil && value > cutoff {
		value = cutoff + 1
	}
	return value, err
}

// LevenshteinNormalizedScore returns normalized Levenshtein similarity.
func LevenshteinNormalizedScore(query, target string) (float64, error) {
	return Score(query, target, ScorerLevenshteinNormalized, nil)
}

// LevenshteinNormalizedScoreCutoff returns zero below cutoff.
func LevenshteinNormalizedScoreCutoff(query, target string, cutoff float64) (float64, error) {
	value, err := LevenshteinNormalizedScore(query, target)
	return similarityCutoff(value, err, cutoff)
}

// DamerauLevenshteinScore returns optimal-string-alignment distance.
func DamerauLevenshteinScore(query, target string) (int64, error) {
	return integerScore(query, target, ScorerDamerauLevenshtein, nil)
}

// OSAScore is an explicit alias for DamerauLevenshteinScore.
func OSAScore(query, target string) (int64, error) {
	return DamerauLevenshteinScore(query, target)
}

// DamerauLevenshteinNormalizedScore returns normalized OSA similarity.
func DamerauLevenshteinNormalizedScore(query, target string) (float64, error) {
	return Score(query, target, ScorerDamerauLevenshteinNormalized, nil)
}

// TrueDamerauLevenshteinScore returns unrestricted Damerau-Levenshtein distance.
func TrueDamerauLevenshteinScore(query, target string) (int64, error) {
	return integerScore(query, target, ScorerTrueDamerauLevenshtein, nil)
}

// TrueDamerauLevenshteinNormalizedScore returns its normalized similarity.
func TrueDamerauLevenshteinNormalizedScore(query, target string) (float64, error) {
	return Score(query, target, ScorerTrueDamerauLevenshteinNormalized, nil)
}

// IndelScore returns insertion/deletion distance.
func IndelScore(query, target string) (int64, error) {
	return integerScore(query, target, ScorerIndel, nil)
}

// IndelScoreCutoff returns cutoff+1 when the distance exceeds cutoff.
func IndelScoreCutoff(query, target string, cutoff int64) (int64, error) {
	if cutoff < 0 {
		return 0, fmt.Errorf("stridealign: score cutoff must be non-negative")
	}
	value, err := IndelScore(query, target)
	if err == nil && value > cutoff {
		value = cutoff + 1
	}
	return value, err
}

// IndelNormalizedScore returns normalized indel similarity.
func IndelNormalizedScore(query, target string) (float64, error) {
	return Score(query, target, ScorerIndelNormalized, nil)
}

// IndelNormalizedScoreCutoff returns zero below cutoff.
func IndelNormalizedScoreCutoff(query, target string, cutoff float64) (float64, error) {
	value, err := IndelNormalizedScore(query, target)
	return similarityCutoff(value, err, cutoff)
}

func similarityCutoff(value float64, err error, cutoff float64) (float64, error) {
	if cutoff < 0 || cutoff > 1 {
		return 0, fmt.Errorf("stridealign: score cutoff must be between 0 and 1")
	}
	if err == nil && value < cutoff {
		value = 0
	}
	return value, err
}

// HammingScore returns Hamming distance and rejects unequal-length inputs.
func HammingScore(query, target string) (int64, error) {
	return integerScore(query, target, ScorerHamming, nil)
}

// HammingNormalizedScore returns normalized Hamming similarity.
func HammingNormalizedScore(query, target string) (float64, error) {
	return Score(query, target, ScorerHammingNormalized, nil)
}

// JaroSimilarity returns Jaro similarity.
func JaroSimilarity(query, target string) (float64, error) {
	return Score(query, target, ScorerJaro, nil)
}

// JaroWinklerSimilarity returns Jaro-Winkler similarity. Pass nil for defaults.
func JaroWinklerSimilarity(query, target string, options *ScoreOptions) (float64, error) {
	return Score(query, target, ScorerJaroWinkler, options)
}

// SmithWatermanScore returns local-alignment score. Pass nil for defaults.
func SmithWatermanScore(query, target string, options *ScoreOptions) (int64, error) {
	return integerScore(query, target, ScorerSmithWaterman, options)
}

// SmithWatermanFarrarScore is retained as the canonical API spelling; it uses
// the same selected native Smith-Waterman kernel.
func SmithWatermanFarrarScore(query, target string, options *ScoreOptions) (int64, error) {
	return SmithWatermanScore(query, target, options)
}

// SmithWatermanNormalizedScore returns normalized local-alignment similarity.
func SmithWatermanNormalizedScore(query, target string, options *ScoreOptions) (float64, error) {
	return Score(query, target, ScorerSmithWatermanNormalized, options)
}

func SmithWatermanFarrarNormalizedScore(query, target string, options *ScoreOptions) (float64, error) {
	return SmithWatermanNormalizedScore(query, target, options)
}

// NeedlemanWunschScore returns global-alignment score. Pass nil for defaults.
func NeedlemanWunschScore(query, target string, options *ScoreOptions) (int64, error) {
	return integerScore(query, target, ScorerNeedlemanWunsch, options)
}

// NeedlemanWunschNormalizedScore returns normalized global-alignment similarity.
func NeedlemanWunschNormalizedScore(query, target string, options *ScoreOptions) (float64, error) {
	return Score(query, target, ScorerNeedlemanWunschNormalized, options)
}
