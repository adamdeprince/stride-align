package stridealign

import "fmt"

// Match is one ranked target from a one-to-many comparison. Index is zero-based.
type Match struct {
	Target string
	Score  float64
	Index  int
}

// PairMatch is one ranked cell from a query-by-target distance matrix. Indices
// are zero-based.
type PairMatch struct {
	Query       string
	Target      string
	Score       float64
	QueryIndex  int
	TargetIndex int
}

// QueryMatches contains the independent top-k result for one query.
type QueryMatches struct {
	Query   string
	Index   int
	Matches []Match
}

// Scores compares one query with every target in a single native call.
func Scores(query string, targets []string, scorer Scorer, options *ScoreOptions) ([]float64, error) {
	return nativeScores(query, targets, scorer, options)
}

// TopK returns the best k targets, with stable input-order tie breaking.
func TopK(query string, targets []string, scorer Scorer, k int, options *ScoreOptions) ([]Match, error) {
	return nativeTopK(query, targets, scorer, k, options, true)
}

// Best returns the best target. The boolean is false when targets is empty or
// no target is valid (for example, unequal-length Hamming inputs).
func Best(query string, targets []string, scorer Scorer, options *ScoreOptions) (Match, bool, error) {
	matches, err := TopK(query, targets, scorer, 1, options)
	if err != nil || len(matches) == 0 {
		return Match{}, false, err
	}
	return matches[0], true, nil
}

// CDist returns a row-major query-by-target score matrix. The outer dimension
// is len(queries); each row has len(targets) values.
func CDist(queries, targets []string, scorer Scorer, options *ScoreOptions) ([][]float64, error) {
	return nativeCDist(queries, targets, scorer, options)
}

// CDistAboveThreshold streams only cells meeting a normalized-similarity
// threshold out of the native matrix analysis.
func CDistAboveThreshold(queries, targets []string, scorer Scorer, threshold float64, options *ScoreOptions) ([]PairMatch, error) {
	return nativeCDistAboveThreshold(queries, targets, scorer, threshold, options)
}

// CDistTopK returns the best k cells across the entire query-by-target matrix.
// It is distinct from CDistTopKPerQuery, which returns up to k cells per query.
func CDistTopK(queries, targets []string, scorer Scorer, k int, rejectDuplicates bool, options *ScoreOptions) ([]PairMatch, error) {
	return nativeCDistTopK(queries, targets, scorer, k, rejectDuplicates, options)
}

// CDistTopKPerQuery returns an independent top-k target list for every query.
func CDistTopKPerQuery(queries, targets []string, scorer Scorer, k int, options *ScoreOptions) ([]QueryMatches, error) {
	return nativeCDistTopKPerQuery(queries, targets, scorer, k, options)
}

// Extract is an alias for TopK.
func Extract(query string, targets []string, scorer Scorer, k int, options *ScoreOptions) ([]Match, error) {
	return TopK(query, targets, scorer, k, options)
}

// ExtractBest is an alias for Best.
func ExtractBest(query string, targets []string, scorer Scorer, options *ScoreOptions) (Match, bool, error) {
	return Best(query, targets, scorer, options)
}

func scorerScores(query string, targets []string, scorer Scorer, options *ScoreOptions) ([]float64, error) {
	return Scores(query, targets, scorer, options)
}

func scorerTopK(query string, targets []string, scorer Scorer, k int, options *ScoreOptions) ([]Match, error) {
	return TopK(query, targets, scorer, k, options)
}

func scorerBest(query string, targets []string, scorer Scorer, options *ScoreOptions) (Match, bool, error) {
	return Best(query, targets, scorer, options)
}

// LevenshteinScores compares one query with many targets.
func LevenshteinScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerLevenshtein, nil)
}

func LevenshteinScoresCutoff(query string, targets []string, cutoff int64) ([]float64, error) {
	if cutoff < 0 {
		return nil, fmt.Errorf("stridealign: score cutoff must be non-negative")
	}
	values, err := LevenshteinScores(query, targets)
	if err != nil {
		return nil, err
	}
	for index, value := range values {
		if value > float64(cutoff) {
			values[index] = float64(cutoff + 1)
		}
	}
	return values, nil
}

func LevenshteinNormalizedScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerLevenshteinNormalized, nil)
}

func LevenshteinNormalizedScoresCutoff(query string, targets []string, cutoff float64) ([]float64, error) {
	return similarityScoresCutoff(query, targets, ScorerLevenshteinNormalized, cutoff)
}

func DamerauLevenshteinScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerDamerauLevenshtein, nil)
}

func DamerauLevenshteinNormalizedScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerDamerauLevenshteinNormalized, nil)
}

func TrueDamerauLevenshteinScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerTrueDamerauLevenshtein, nil)
}

func TrueDamerauLevenshteinNormalizedScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerTrueDamerauLevenshteinNormalized, nil)
}

func IndelScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerIndel, nil)
}

func IndelScoresCutoff(query string, targets []string, cutoff int64) ([]float64, error) {
	if cutoff < 0 {
		return nil, fmt.Errorf("stridealign: score cutoff must be non-negative")
	}
	values, err := IndelScores(query, targets)
	if err != nil {
		return nil, err
	}
	for index, value := range values {
		if value > float64(cutoff) {
			values[index] = float64(cutoff + 1)
		}
	}
	return values, nil
}

func IndelNormalizedScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerIndelNormalized, nil)
}

func IndelNormalizedScoresCutoff(query string, targets []string, cutoff float64) ([]float64, error) {
	return similarityScoresCutoff(query, targets, ScorerIndelNormalized, cutoff)
}

func similarityScoresCutoff(query string, targets []string, scorer Scorer, cutoff float64) ([]float64, error) {
	if cutoff < 0 || cutoff > 1 {
		return nil, fmt.Errorf("stridealign: score cutoff must be between 0 and 1")
	}
	values, err := Scores(query, targets, scorer, nil)
	if err != nil {
		return nil, err
	}
	for index, value := range values {
		if value < cutoff {
			values[index] = 0
		}
	}
	return values, nil
}

func HammingScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerHamming, nil)
}

func HammingNormalizedScores(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerHammingNormalized, nil)
}

func JaroSimilarities(query string, targets []string) ([]float64, error) {
	return scorerScores(query, targets, ScorerJaro, nil)
}

func JaroWinklerSimilarities(query string, targets []string, options *ScoreOptions) ([]float64, error) {
	return scorerScores(query, targets, ScorerJaroWinkler, options)
}

func SmithWatermanScores(query string, targets []string, options *ScoreOptions) ([]float64, error) {
	return scorerScores(query, targets, ScorerSmithWaterman, options)
}

func SmithWatermanFarrarScores(query string, targets []string, options *ScoreOptions) ([]float64, error) {
	return SmithWatermanScores(query, targets, options)
}

func SmithWatermanNormalizedScores(query string, targets []string, options *ScoreOptions) ([]float64, error) {
	return scorerScores(query, targets, ScorerSmithWatermanNormalized, options)
}

func SmithWatermanFarrarNormalizedScores(query string, targets []string, options *ScoreOptions) ([]float64, error) {
	return SmithWatermanNormalizedScores(query, targets, options)
}

func NeedlemanWunschScores(query string, targets []string, options *ScoreOptions) ([]float64, error) {
	return scorerScores(query, targets, ScorerNeedlemanWunsch, options)
}

func NeedlemanWunschNormalizedScores(query string, targets []string, options *ScoreOptions) ([]float64, error) {
	return scorerScores(query, targets, ScorerNeedlemanWunschNormalized, options)
}

func LevenshteinTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerLevenshtein, k, nil)
}

func LevenshteinNormalizedTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerLevenshteinNormalized, k, nil)
}

func DamerauLevenshteinTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerDamerauLevenshtein, k, nil)
}

func DamerauLevenshteinNormalizedTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerDamerauLevenshteinNormalized, k, nil)
}

func TrueDamerauLevenshteinTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerTrueDamerauLevenshtein, k, nil)
}

func TrueDamerauLevenshteinNormalizedTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerTrueDamerauLevenshteinNormalized, k, nil)
}

func IndelTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerIndel, k, nil)
}

func IndelNormalizedTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerIndelNormalized, k, nil)
}

func HammingTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerHamming, k, nil)
}

func HammingNormalizedTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerHammingNormalized, k, nil)
}

func JaroTopK(query string, targets []string, k int) ([]Match, error) {
	return scorerTopK(query, targets, ScorerJaro, k, nil)
}

func JaroWinklerTopK(query string, targets []string, k int, options *ScoreOptions) ([]Match, error) {
	return scorerTopK(query, targets, ScorerJaroWinkler, k, options)
}

func SmithWatermanTopK(query string, targets []string, k int, options *ScoreOptions) ([]Match, error) {
	return scorerTopK(query, targets, ScorerSmithWaterman, k, options)
}

func LevenshteinBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerLevenshtein, nil)
}

func LevenshteinNormalizedBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerLevenshteinNormalized, nil)
}

func DamerauLevenshteinBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerDamerauLevenshtein, nil)
}

func DamerauLevenshteinNormalizedBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerDamerauLevenshteinNormalized, nil)
}

func TrueDamerauLevenshteinBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerTrueDamerauLevenshtein, nil)
}

func TrueDamerauLevenshteinNormalizedBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerTrueDamerauLevenshteinNormalized, nil)
}

func IndelBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerIndel, nil)
}

func IndelNormalizedBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerIndelNormalized, nil)
}

func HammingBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerHamming, nil)
}

func HammingNormalizedBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerHammingNormalized, nil)
}

func JaroBest(query string, targets []string) (Match, bool, error) {
	return scorerBest(query, targets, ScorerJaro, nil)
}

func JaroWinklerBest(query string, targets []string, options *ScoreOptions) (Match, bool, error) {
	return scorerBest(query, targets, ScorerJaroWinkler, options)
}

func SmithWatermanBest(query string, targets []string, options *ScoreOptions) (Match, bool, error) {
	return scorerBest(query, targets, ScorerSmithWaterman, options)
}

// LevenshteinScorer caches one query for repeated one-to-one or batch calls.
type LevenshteinScorer struct{ Query string }

func NewLevenshteinScorer(query string) LevenshteinScorer {
	return LevenshteinScorer{Query: query}
}

func (scorer LevenshteinScorer) Distance(target string) (int64, error) {
	return LevenshteinScore(scorer.Query, target)
}

func (scorer LevenshteinScorer) DistanceCutoff(target string, cutoff int64) (int64, error) {
	return LevenshteinScoreCutoff(scorer.Query, target, cutoff)
}

func (scorer LevenshteinScorer) NormalizedDistance(target string) (float64, error) {
	return LevenshteinNormalizedScore(scorer.Query, target)
}

func (scorer LevenshteinScorer) NormalizedDistanceCutoff(target string, cutoff float64) (float64, error) {
	return LevenshteinNormalizedScoreCutoff(scorer.Query, target, cutoff)
}

func (scorer LevenshteinScorer) Distances(targets []string) ([]float64, error) {
	return LevenshteinScores(scorer.Query, targets)
}

func (scorer LevenshteinScorer) DistancesCutoff(targets []string, cutoff int64) ([]float64, error) {
	return LevenshteinScoresCutoff(scorer.Query, targets, cutoff)
}

func (scorer LevenshteinScorer) NormalizedDistances(targets []string) ([]float64, error) {
	return LevenshteinNormalizedScores(scorer.Query, targets)
}

func (scorer LevenshteinScorer) NormalizedDistancesCutoff(targets []string, cutoff float64) ([]float64, error) {
	return LevenshteinNormalizedScoresCutoff(scorer.Query, targets, cutoff)
}
