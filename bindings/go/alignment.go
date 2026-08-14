package stridealign

import "fmt"

// AlignmentPath describes an optimal alignment. Positions are zero-based
// half-open Unicode-code-point offsets. Operations use =, X, I, and D.
type AlignmentPath struct {
	Score         int64
	QueryStart    int
	QueryEnd      int
	TargetStart   int
	TargetEnd     int
	Operations    string
	CIGAR         string
	Matches       int
	Mismatches    int
	Insertions    int
	Deletions     int
	AlignedLength int
	AlignedQuery  string
	AlignedTarget string
}

// SmithWatermanPath returns a local-alignment traceback.
func SmithWatermanPath(query, target string, options *ScoreOptions) (AlignmentPath, error) {
	return nativePath(true, query, target, options)
}

// NeedlemanWunschPath returns a global-alignment traceback.
func NeedlemanWunschPath(query, target string, options *ScoreOptions) (AlignmentPath, error) {
	return nativePath(false, query, target, options)
}

func SmithWatermanPathInfo(query, target string, options *ScoreOptions) (AlignmentPath, error) {
	return SmithWatermanPath(query, target, options)
}

func NeedlemanWunschPathInfo(query, target string, options *ScoreOptions) (AlignmentPath, error) {
	return NeedlemanWunschPath(query, target, options)
}

func SmithWatermanCIGAR(query, target string, options *ScoreOptions) (string, error) {
	path, err := SmithWatermanPath(query, target, options)
	return path.CIGAR, err
}

func SmithWatermanTraceCIGAR(query, target string, options *ScoreOptions) (string, error) {
	return SmithWatermanCIGAR(query, target, options)
}

func SmithWatermanTradeCIGAR(query, target string, options *ScoreOptions) (string, error) {
	return SmithWatermanCIGAR(query, target, options)
}

func NeedlemanWunschCIGAR(query, target string, options *ScoreOptions) (string, error) {
	path, err := NeedlemanWunschPath(query, target, options)
	return path.CIGAR, err
}

func NeedlemanWunschTraceCIGAR(query, target string, options *ScoreOptions) (string, error) {
	return NeedlemanWunschCIGAR(query, target, options)
}

func NeedlemanWunschTradeCIGAR(query, target string, options *ScoreOptions) (string, error) {
	return NeedlemanWunschCIGAR(query, target, options)
}

// AlignmentVariant selects a cached-query alignment scorer.
type AlignmentVariant int

const (
	AlignmentSmithWaterman AlignmentVariant = iota
	AlignmentNeedlemanWunsch
	AlignmentSmithWatermanFarrar
)

// AlignmentScores caches a query and scoring configuration for repeated batch calls.
type AlignmentScores struct {
	Query   string
	Variant AlignmentVariant
	Options ScoreOptions
}

func NewAlignmentScores(query string, variant AlignmentVariant, options *ScoreOptions) AlignmentScores {
	resolved := DefaultScoreOptions()
	if options != nil {
		resolved = *options
	}
	return AlignmentScores{Query: query, Variant: variant, Options: resolved}
}

func (scores AlignmentScores) Compare(targets []string) ([]float64, error) {
	var scorer Scorer
	switch scores.Variant {
	case AlignmentSmithWaterman, AlignmentSmithWatermanFarrar:
		scorer = ScorerSmithWaterman
	case AlignmentNeedlemanWunsch:
		scorer = ScorerNeedlemanWunsch
	default:
		return nil, fmt.Errorf("stridealign: unknown alignment variant %d", scores.Variant)
	}
	return Scores(scores.Query, targets, scorer, &scores.Options)
}
