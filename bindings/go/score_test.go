package stridealign

import (
	"math"
	"testing"
)

func closeEnough(left, right float64) bool { return math.Abs(left-right) <= 1e-12 }

func requireScore(t *testing.T, value float64, err error, expected float64) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	if !closeEnough(value, expected) {
		t.Fatalf("score = %v, want %v", value, expected)
	}
}

func TestVersionAndBackend(t *testing.T) {
	if Version() != "0.6.0" {
		t.Fatalf("Version() = %q", Version())
	}
	if Backend() == "" || !BackendIsAvailable(Backend()) {
		t.Fatalf("invalid backend %q", Backend())
	}
	if len(AvailableBackends()) != 1 {
		t.Fatalf("AvailableBackends() = %#v", AvailableBackends())
	}
}

func TestScalarScorerCatalog(t *testing.T) {
	distance, err := LevenshteinScore("kitten", "sitting")
	if err != nil || distance != 3 {
		t.Fatalf("LevenshteinScore = %d, %v", distance, err)
	}
	normalized, err := LevenshteinNormalizedScore("foobar", "foobaz")
	requireScore(t, normalized, err, 5.0/6.0)

	distance, err = DamerauLevenshteinScore("abcd", "acbd")
	if err != nil || distance != 1 {
		t.Fatalf("DamerauLevenshteinScore = %d, %v", distance, err)
	}
	distance, err = TrueDamerauLevenshteinScore("CA", "ABC")
	if err != nil || distance != 2 {
		t.Fatalf("TrueDamerauLevenshteinScore = %d, %v", distance, err)
	}
	distance, err = IndelScore("lewenstein", "levenshtein")
	if err != nil || distance != 3 {
		t.Fatalf("IndelScore = %d, %v", distance, err)
	}
	distance, err = HammingScore("karolin", "kathrin")
	if err != nil || distance != 3 {
		t.Fatalf("HammingScore = %d, %v", distance, err)
	}
	jaro, err := JaroSimilarity("MARTHA", "MARHTA")
	requireScore(t, jaro, err, 0.9444444444444445)
	winkler, err := JaroWinklerSimilarity("MARTHA", "MARHTA", nil)
	requireScore(t, winkler, err, 0.9611111111111111)

	sw, err := SmithWatermanScore("ACCGT", "CCG", nil)
	if err != nil || sw != 6 {
		t.Fatalf("SmithWatermanScore = %d, %v", sw, err)
	}
	nw, err := NeedlemanWunschScore("ACGT", "ACCT", nil)
	if err != nil || nw != 5 {
		t.Fatalf("NeedlemanWunschScore = %d, %v", nw, err)
	}
}

func TestUnicodeAndEmbeddedNUL(t *testing.T) {
	tests := []struct {
		left, right string
		want        int64
	}{
		{"café", "cafe", 1},
		{"🎉🎈", "🎈🎉", 2},
		{"a\x00b", "a\x00c", 1},
		{"𐐷文🙂", "𐐷字🙂", 1},
	}
	for _, test := range tests {
		got, err := LevenshteinScore(test.left, test.right)
		if err != nil || got != test.want {
			t.Errorf("LevenshteinScore(%q,%q) = %d, %v; want %d", test.left, test.right, got, err, test.want)
		}
	}
	if _, err := LevenshteinScore("\xff", "x"); err == nil {
		t.Fatal("invalid UTF-8 was accepted")
	}
}

func TestBatchAndRankedAPIs(t *testing.T) {
	targets := []string{"sitting", "kitten", "bitten", "mittens"}
	scores, err := LevenshteinScores("kitten", targets)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{3, 0, 1, 2}
	for index := range want {
		if scores[index] != want[index] {
			t.Fatalf("scores = %v, want %v", scores, want)
		}
	}
	top, err := LevenshteinTopK("kitten", targets, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(top) != 2 || top[0].Index != 1 || top[1].Index != 2 {
		t.Fatalf("top = %#v", top)
	}
	best, ok, err := LevenshteinBest("kitten", targets)
	if err != nil || !ok || best.Index != 1 || best.Target != "kitten" {
		t.Fatalf("best = %#v, %v, %v", best, ok, err)
	}
	if _, ok, err := Best("x", nil, ScorerJaro, nil); err != nil || ok {
		t.Fatalf("empty Best returned ok=%v err=%v", ok, err)
	}
}

func TestCDistShapesAndTopKForms(t *testing.T) {
	queries := []string{"cat", "dog"}
	targets := []string{"cat", "cot", "dog"}
	dense, err := CDist(queries, targets, ScorerLevenshtein, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float64{{0, 1, 3}, {3, 2, 0}}
	for row := range want {
		for column := range want[row] {
			if dense[row][column] != want[row][column] {
				t.Fatalf("dense = %#v, want %#v", dense, want)
			}
		}
	}
	filtered, err := CDistAboveThreshold(queries[:1], targets, ScorerLevenshteinNormalized, 0.6, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 2 || filtered[0].TargetIndex != 0 || filtered[1].TargetIndex != 1 {
		t.Fatalf("filtered = %#v", filtered)
	}
	global, err := CDistTopK(queries, targets, ScorerJaro, 2, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(global) != 2 {
		t.Fatalf("global top-k = %#v", global)
	}
	for _, match := range global {
		if match.Query == match.Target {
			t.Fatalf("duplicate pair was retained: %#v", match)
		}
	}
	perQuery, err := CDistTopKPerQuery(queries, targets, ScorerLevenshteinNormalized, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(perQuery) != 2 || perQuery[0].Matches[0].Index != 0 || perQuery[1].Matches[0].Index != 2 {
		t.Fatalf("per-query top-k = %#v", perQuery)
	}
	if _, err := CDistTopK(queries, targets, ScorerLevenshtein, 1, false, nil); err == nil {
		t.Fatal("distance scorer accepted by CDistTopK")
	}
}

func TestScorerAliasesAndPreparedQuery(t *testing.T) {
	for name, want := range map[string]Scorer{
		"levenshtein":    ScorerLevenshtein,
		"osa":            ScorerDamerauLevenshtein,
		"smith-waterman": ScorerSmithWaterman,
		"global":         ScorerNeedlemanWunsch,
	} {
		got, err := ParseScorer(name)
		if err != nil || got != want {
			t.Errorf("ParseScorer(%q) = %v, %v", name, got, err)
		}
	}
	prepared := NewLevenshteinScorer("god")
	distance, err := prepared.Distance("good")
	if err != nil || distance != 1 {
		t.Fatalf("prepared distance = %d, %v", distance, err)
	}
	distances, err := prepared.Distances([]string{"good", "god", "god!"})
	if err != nil || len(distances) != 3 || distances[0] != 1 || distances[1] != 0 || distances[2] != 1 {
		t.Fatalf("prepared distances = %v, %v", distances, err)
	}
}

func TestScoreCutoffs(t *testing.T) {
	distance, err := LevenshteinScoreCutoff("kitten", "sitting", 1)
	if err != nil || distance != 2 {
		t.Fatalf("LevenshteinScoreCutoff = %d, %v", distance, err)
	}
	similarity, err := IndelNormalizedScoreCutoff("hello", "world", 0.5)
	if err != nil || similarity != 0 {
		t.Fatalf("IndelNormalizedScoreCutoff = %v, %v", similarity, err)
	}
	values, err := LevenshteinScoresCutoff("abc", []string{"abc", "axc", "xyz"}, 1)
	if err != nil || len(values) != 3 || values[0] != 0 || values[1] != 1 || values[2] != 2 {
		t.Fatalf("LevenshteinScoresCutoff = %#v, %v", values, err)
	}
}
