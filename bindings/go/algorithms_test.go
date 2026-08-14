package stridealign

import (
	"strings"
	"testing"
)

func TestLCSAndNGrams(t *testing.T) {
	length, err := LCSLength("ABCBDAB", "BDCABA")
	if err != nil || length != 4 {
		t.Fatalf("LCSLength = %d, %v", length, err)
	}
	substring, err := LCSSubstring("xabxac", "abcabxabcd")
	if err != nil || substring != "abxa" {
		t.Fatalf("LCSSubstring = %q, %v", substring, err)
	}
	length, err = LCSSubstringLength("xabxac", "abcabxabcd")
	if err != nil || length != 4 {
		t.Fatalf("LCSSubstringLength = %d, %v", length, err)
	}
	substrings, err := LCSSubstrings("xabxac", []string{"abcabxabcd", "zzabxayy"})
	if err != nil || len(substrings) != 2 || substrings[0] != "abxa" || substrings[1] != "abxa" {
		t.Fatalf("LCSSubstrings = %#v, %v", substrings, err)
	}

	for name, function := range map[string]func(string, string, int) (float64, error){
		"jaccard": Jaccard,
		"dice":    Dice,
		"cosine":  Cosine,
		"overlap": Overlap,
	} {
		identical, err := function("night", "night", 2)
		if err != nil || identical != 1 {
			t.Errorf("%s identical = %v, %v", name, identical, err)
		}
		disjoint, err := function("abc", "xyz", 2)
		if err != nil || disjoint != 0 {
			t.Errorf("%s disjoint = %v, %v", name, disjoint, err)
		}
	}
	values, err := JaccardSimilarities("night", []string{"night", "nacht", "xyz"}, 2)
	if err != nil || len(values) != 3 || values[0] != 1 || values[2] != 0 {
		t.Fatalf("JaccardSimilarities = %v, %v", values, err)
	}
}

func TestFuzzyRatiosAndProcessor(t *testing.T) {
	partial, err := PartialRatio("this is a test", "this is a test!")
	if err != nil || partial != 1 {
		t.Fatalf("PartialRatio = %v, %v", partial, err)
	}
	sorted, err := TokenSortRatio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
	if err != nil || sorted != 1 {
		t.Fatalf("TokenSortRatio = %v, %v", sorted, err)
	}
	set, err := TokenSetRatio("fuzzy was a bear", "fuzzy fuzzy was a bear")
	if err != nil || set != 1 {
		t.Fatalf("TokenSetRatio = %v, %v", set, err)
	}
	processed, err := WRatioWithProcessor(" MARTHA ", "martha", func(value string) (string, error) {
		return strings.ToLower(strings.TrimSpace(value)), nil
	})
	if err != nil || processed != 1 {
		t.Fatalf("processed WRatio = %v, %v", processed, err)
	}
	ratcliff, err := RatcliffObershelpSimilarity("abcd", "abxcd")
	if err != nil || !closeEnough(ratcliff, 8.0/9.0) {
		t.Fatalf("RatcliffObershelpSimilarity = %v, %v", ratcliff, err)
	}
	empty, err := WRatio("", "")
	if err != nil || empty != 1 {
		t.Fatalf("empty WRatio = %v, %v", empty, err)
	}
	batch, err := TokenSortRatios("fuzzy wuzzy", []string{"wuzzy fuzzy", "different"}, nil)
	if err != nil || len(batch) != 2 || batch[0] != 1 {
		t.Fatalf("TokenSortRatios = %#v, %v", batch, err)
	}
}

func TestMongeElkanNamedAndCustomInner(t *testing.T) {
	value, err := MongeElkan("jane doe", "jane roe", "jaro", false)
	if err != nil || value <= 0 || value > 1 {
		t.Fatalf("MongeElkan = %v, %v", value, err)
	}
	custom, err := MongeElkanWith(
		"ALPHA BETA", "alpha zeta",
		func(left, right string) (float64, error) { return JaroSimilarity(left, right) },
		func(value string) (string, error) { return strings.ToLower(value), nil },
		true,
	)
	if err != nil || custom <= 0 || custom > 1 {
		t.Fatalf("MongeElkanWith = %v, %v", custom, err)
	}
	empty, err := MongeElkan("", "", "jaro", true)
	if err != nil || empty != 1 {
		t.Fatalf("empty MongeElkan = %v, %v", empty, err)
	}
}
