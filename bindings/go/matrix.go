package stridealign

import (
	"bufio"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// SubstitutionMatrix is an immutable-by-convention row-major int8 scoring grid.
// Unknown symbols encode as WildcardIndex.
type SubstitutionMatrix struct {
	Name           string
	Alphabet       string
	WildcardIndex  int
	Values         []int8
	GapScore       int64
	GapOpenScore   int64
	GapExtendScore int64
	HasAffineGaps  bool
}

// MatrixOptions controls construction of a custom substitution matrix.
type MatrixOptions struct {
	Wildcard       rune
	GapScore       int64
	GapOpenScore   int64
	GapExtendScore int64
	AffineGaps     bool
}

func DefaultMatrixOptions() MatrixOptions {
	return MatrixOptions{Wildcard: 'X', GapScore: -4}
}

// NewSubstitutionMatrix validates and snapshots a custom score grid.
func NewSubstitutionMatrix(name, alphabet string, scores [][]int, options *MatrixOptions) (*SubstitutionMatrix, error) {
	resolved := DefaultMatrixOptions()
	if options != nil {
		resolved = *options
	}
	symbols := []rune(alphabet)
	if len(symbols) == 0 || len(symbols) > 256 {
		return nil, fmt.Errorf("stridealign: matrix alphabet must contain 1 through 256 symbols")
	}
	seen := make(map[rune]struct{}, len(symbols))
	wildcardIndex := -1
	for index, symbol := range symbols {
		if _, exists := seen[symbol]; exists {
			return nil, fmt.Errorf("stridealign: matrix alphabet contains duplicate symbol %q", symbol)
		}
		seen[symbol] = struct{}{}
		if symbol == resolved.Wildcard {
			wildcardIndex = index
		}
	}
	if wildcardIndex < 0 {
		return nil, fmt.Errorf("stridealign: wildcard %q is absent from matrix alphabet", resolved.Wildcard)
	}
	if len(scores) != len(symbols) {
		return nil, fmt.Errorf("stridealign: matrix row count does not match alphabet")
	}
	values := make([]int8, 0, len(symbols)*len(symbols))
	for _, row := range scores {
		if len(row) != len(symbols) {
			return nil, fmt.Errorf("stridealign: matrix column count does not match alphabet")
		}
		for _, score := range row {
			if score < -128 || score > 127 {
				return nil, fmt.Errorf("stridealign: matrix score %d does not fit int8", score)
			}
			values = append(values, int8(score))
		}
	}
	return &SubstitutionMatrix{
		Name: name, Alphabet: alphabet, WildcardIndex: wildcardIndex,
		Values: values, GapScore: resolved.GapScore,
		GapOpenScore:   resolved.GapOpenScore,
		GapExtendScore: resolved.GapExtendScore,
		HasAffineGaps:  resolved.AffineGaps,
	}, nil
}

// AvailableMatrices returns embedded protein, nucleotide, text, and keyboard matrices.
func AvailableMatrices() ([]string, error) { return nativeMatrixAvailable() }

// BuiltinMatrix loads a private snapshot of an embedded matrix by name.
func BuiltinMatrix(name string) (*SubstitutionMatrix, error) { return nativeMatrix(name) }

// Encode maps a UTF-8 sequence to zero-based alphabet indices.
func (matrix *SubstitutionMatrix) Encode(sequence string) ([]uint16, error) {
	return nativeMatrixEncode(matrix, sequence)
}

// MatrixGapOptions overrides the matrix's default linear or affine gap scores.
type MatrixGapOptions struct {
	GapOpenScore   int64
	GapExtendScore int64
}

func (matrix *SubstitutionMatrix) resolveGaps(options *MatrixGapOptions) (int64, int64) {
	if options != nil {
		return options.GapOpenScore, options.GapExtendScore
	}
	if matrix.HasAffineGaps {
		return matrix.GapOpenScore, matrix.GapExtendScore
	}
	return matrix.GapScore, matrix.GapScore
}

// ScoreStepLimit returns the largest absolute matrix or gap transition.
func (matrix *SubstitutionMatrix) ScoreStepLimit(options *MatrixGapOptions) (int64, error) {
	open, extend := matrix.resolveGaps(options)
	return nativeMatrixStepLimit(matrix, open, extend)
}

// Score is an alias for SmithWatermanScore, matching R's substitution_matrix_score.
func (matrix *SubstitutionMatrix) Score(query, target string, options *MatrixGapOptions) (int64, error) {
	return matrix.SmithWatermanScore(query, target, options)
}

func (matrix *SubstitutionMatrix) SmithWatermanScore(query, target string, options *MatrixGapOptions) (int64, error) {
	open, extend := matrix.resolveGaps(options)
	return nativeMatrixScore(true, matrix, query, target, open, extend)
}

func (matrix *SubstitutionMatrix) NeedlemanWunschScore(query, target string, options *MatrixGapOptions) (int64, error) {
	open, extend := matrix.resolveGaps(options)
	return nativeMatrixScore(false, matrix, query, target, open, extend)
}

func (matrix *SubstitutionMatrix) SmithWatermanPath(query, target string, options *MatrixGapOptions) (AlignmentPath, error) {
	open, extend := matrix.resolveGaps(options)
	return nativeMatrixPath(true, matrix, query, target, open, extend)
}

func (matrix *SubstitutionMatrix) NeedlemanWunschPath(query, target string, options *MatrixGapOptions) (AlignmentPath, error) {
	open, extend := matrix.resolveGaps(options)
	return nativeMatrixPath(false, matrix, query, target, open, extend)
}

func (matrix *SubstitutionMatrix) SmithWatermanCIGAR(query, target string, options *MatrixGapOptions) (string, error) {
	path, err := matrix.SmithWatermanPath(query, target, options)
	return path.CIGAR, err
}

func (matrix *SubstitutionMatrix) NeedlemanWunschCIGAR(query, target string, options *MatrixGapOptions) (string, error) {
	path, err := matrix.NeedlemanWunschPath(query, target, options)
	return path.CIGAR, err
}

// Transpose returns a transposed private snapshot.
func (matrix *SubstitutionMatrix) Transpose(name string) (*SubstitutionMatrix, error) {
	if matrix == nil {
		return nil, fmt.Errorf("stridealign: matrix must not be nil")
	}
	size := utf8.RuneCountInString(matrix.Alphabet)
	if size == 0 || len(matrix.Values) != size*size {
		return nil, fmt.Errorf("stridealign: invalid matrix dimensions")
	}
	values := make([]int8, len(matrix.Values))
	for row := 0; row < size; row++ {
		for column := 0; column < size; column++ {
			values[column*size+row] = matrix.Values[row*size+column]
		}
	}
	if name == "" {
		name = matrix.Name + ".T"
	}
	return &SubstitutionMatrix{
		Name: name, Alphabet: matrix.Alphabet, WildcardIndex: matrix.WildcardIndex,
		Values: values, GapScore: matrix.GapScore,
		GapOpenScore: matrix.GapOpenScore, GapExtendScore: matrix.GapExtendScore,
		HasAffineGaps: matrix.HasAffineGaps,
	}, nil
}

// NCBIMatrixOptions controls parsing of an NCBI text-format score matrix.
type NCBIMatrixOptions struct {
	Name       string
	Wildcard   rune
	GapOptions MatrixOptions
}

// SubstitutionMatrixFromNCBIText parses an NCBI matrix without external files.
func SubstitutionMatrixFromNCBIText(text string, options *NCBIMatrixOptions) (*SubstitutionMatrix, error) {
	resolved := NCBIMatrixOptions{Name: "<unnamed>", Wildcard: 'X', GapOptions: DefaultMatrixOptions()}
	if options != nil {
		resolved = *options
	}
	if resolved.Wildcard == 0 {
		resolved.Wildcard = 'X'
	}
	var lines [][]string
	scanner := bufio.NewScanner(strings.NewReader(text))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		lines = append(lines, strings.Fields(line))
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(lines) < 2 || len(lines[0]) == 0 {
		return nil, fmt.Errorf("stridealign: NCBI matrix has no header or data rows")
	}
	header := lines[0]
	scores := make([][]int, len(header))
	if len(lines)-1 != len(header) {
		return nil, fmt.Errorf("stridealign: NCBI matrix row count does not match header")
	}
	for rowIndex, row := range lines[1:] {
		if len(row) != len(header)+1 || row[0] != header[rowIndex] {
			return nil, fmt.Errorf("stridealign: NCBI matrix row %d does not match header", rowIndex+1)
		}
		scores[rowIndex] = make([]int, len(header))
		for column, value := range row[1:] {
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return nil, fmt.Errorf("stridealign: non-integer NCBI matrix score %q", value)
			}
			scores[rowIndex][column] = parsed
		}
	}
	matrixOptions := resolved.GapOptions
	matrixOptions.Wildcard = resolved.Wildcard
	return NewSubstitutionMatrix(resolved.Name, strings.Join(header, ""), scores, &matrixOptions)
}

// IdentityMatrix creates a match/mismatch matrix, adding wildcard if needed.
func IdentityMatrix(name, alphabet string, match, mismatch int, wildcard rune, options *MatrixOptions) (*SubstitutionMatrix, error) {
	symbols := []rune(alphabet)
	found := false
	for _, symbol := range symbols {
		if symbol == wildcard {
			found = true
		}
	}
	if !found {
		symbols = append(symbols, wildcard)
	}
	scores := make([][]int, len(symbols))
	wildcardIndex := 0
	for row := range scores {
		scores[row] = make([]int, len(symbols))
		for column := range scores[row] {
			scores[row][column] = mismatch
		}
		scores[row][row] = match
		if symbols[row] == wildcard {
			wildcardIndex = row
		}
	}
	for index := range scores {
		scores[wildcardIndex][index] = mismatch
		scores[index][wildcardIndex] = mismatch
	}
	resolved := DefaultMatrixOptions()
	resolved.GapScore = -1
	if options != nil {
		resolved = *options
	}
	resolved.Wildcard = wildcard
	return NewSubstitutionMatrix(name, string(symbols), scores, &resolved)
}

// ASCIIMatrix creates a full 128-symbol ASCII matrix, including NUL.
func ASCIIMatrix(name string, match, mismatch int, options *MatrixOptions) (*SubstitutionMatrix, error) {
	symbols := make([]rune, 128)
	for index := range symbols {
		symbols[index] = rune(index)
	}
	return IdentityMatrix(name, string(symbols), match, mismatch, rune(127), options)
}

func (matrix *SubstitutionMatrix) matrixCDist(queries, targets []string, local bool, options *MatrixGapOptions) ([][]float64, error) {
	open, extend := matrix.resolveGaps(options)
	return nativeMatrixCDist(local, matrix, queries, targets, open, extend)
}

// CDist computes a matrix-scored query-by-target grid.
func (matrix *SubstitutionMatrix) CDist(queries, targets []string, local bool, options *MatrixGapOptions) ([][]float64, error) {
	return matrix.matrixCDist(queries, targets, local, options)
}

func (matrix *SubstitutionMatrix) CDistAboveThreshold(queries, targets []string, local bool, threshold float64, options *MatrixGapOptions) ([]PairMatch, error) {
	grid, err := matrix.matrixCDist(queries, targets, local, options)
	if err != nil {
		return nil, err
	}
	var result []PairMatch
	for queryIndex, row := range grid {
		for targetIndex, score := range row {
			if score >= threshold {
				result = append(result, PairMatch{Query: queries[queryIndex], Target: targets[targetIndex], Score: score, QueryIndex: queryIndex, TargetIndex: targetIndex})
			}
		}
	}
	return result, nil
}

func (matrix *SubstitutionMatrix) CDistTopK(queries, targets []string, local bool, k int, options *MatrixGapOptions) ([]PairMatch, error) {
	if k < 0 {
		return nil, fmt.Errorf("stridealign: k must be non-negative")
	}
	grid, err := matrix.matrixCDist(queries, targets, local, options)
	if err != nil {
		return nil, err
	}
	result := make([]PairMatch, 0, len(queries)*len(targets))
	for queryIndex, row := range grid {
		for targetIndex, score := range row {
			result = append(result, PairMatch{Query: queries[queryIndex], Target: targets[targetIndex], Score: score, QueryIndex: queryIndex, TargetIndex: targetIndex})
		}
	}
	sort.SliceStable(result, func(left, right int) bool { return result[left].Score > result[right].Score })
	if k < len(result) {
		result = result[:k]
	}
	return result, nil
}

func (matrix *SubstitutionMatrix) CDistTopKPerQuery(queries, targets []string, local bool, k int, options *MatrixGapOptions) ([]QueryMatches, error) {
	if k < 0 {
		return nil, fmt.Errorf("stridealign: k must be non-negative")
	}
	grid, err := matrix.matrixCDist(queries, targets, local, options)
	if err != nil {
		return nil, err
	}
	result := make([]QueryMatches, len(queries))
	for queryIndex, row := range grid {
		matches := make([]Match, len(row))
		for targetIndex, score := range row {
			matches[targetIndex] = Match{Target: targets[targetIndex], Score: score, Index: targetIndex}
		}
		sort.SliceStable(matches, func(left, right int) bool { return matches[left].Score > matches[right].Score })
		if k < len(matches) {
			matches = matches[:k]
		}
		result[queryIndex] = QueryMatches{Query: queries[queryIndex], Index: queryIndex, Matches: matches}
	}
	return result, nil
}
