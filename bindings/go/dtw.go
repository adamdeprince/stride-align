package stridealign

// DTWDistance selects the per-element cost used by dynamic time warping.
type DTWDistance int

const (
	DTWL1 DTWDistance = iota
	DTWL2Squared
)

// DTWOptions controls dynamic time warping. Window is -1 for unrestricted, a
// fraction in (0,1), or an integer-valued radius. ScoreCutoff is -1 to disable.
type DTWOptions struct {
	Window      float64
	Distance    DTWDistance
	ScoreCutoff float64
}

func DefaultDTWOptions() DTWOptions {
	return DTWOptions{Window: -1, Distance: DTWL2Squared, ScoreCutoff: -1}
}

// DTW returns the dynamic-time-warping distance between numeric sequences.
func DTW(query, target []float64, options *DTWOptions) (float64, error) {
	return nativeDTW(query, target, options)
}

// DTWDistances compares one sequence with every target sequence.
func DTWDistances(query []float64, targets [][]float64, options *DTWOptions) ([]float64, error) {
	return nativeDTWDistances(query, targets, options)
}

// DTWInts is the integer-sequence counterpart; L1 is the conventional default.
func DTWInts(query, target []int64, options *DTWOptions) (float64, error) {
	left := floatSequence(query)
	right := floatSequence(target)
	if options == nil {
		resolved := DefaultDTWOptions()
		resolved.Distance = DTWL1
		options = &resolved
	}
	return nativeDTW(left, right, options)
}

// DTWIntsDistances compares one integer sequence with every integer target.
func DTWIntsDistances(query []int64, targets [][]int64, options *DTWOptions) ([]float64, error) {
	left := floatSequence(query)
	right := make([][]float64, len(targets))
	for index, target := range targets {
		right[index] = floatSequence(target)
	}
	if options == nil {
		resolved := DefaultDTWOptions()
		resolved.Distance = DTWL1
		options = &resolved
	}
	return nativeDTWDistances(left, right, options)
}

func floatSequence(input []int64) []float64 {
	output := make([]float64, len(input))
	for index, value := range input {
		output[index] = float64(value)
	}
	return output
}
