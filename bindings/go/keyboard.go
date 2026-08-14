package stridealign

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// KeyboardASCIIAlphabet contains all 128 ASCII code points, including NUL.
var KeyboardASCIIAlphabet = func() string {
	symbols := make([]rune, 128)
	for index := range symbols {
		symbols[index] = rune(index)
	}
	return string(symbols)
}()

const KeyboardASCIIWildcard rune = 127

// KeyboardOptions controls conversion of keyboard confusion counts.
type KeyboardOptions struct {
	Name           string
	Alphabet       string
	Wildcard       rune
	Scale          float64
	MatchMargin    float64
	Floor          *float64
	GapScore       int64
	GapOpenScore   int64
	GapExtendScore int64
	AffineGaps     bool
}

func DefaultKeyboardOptions() KeyboardOptions {
	return KeyboardOptions{
		Name: "KEYBOARD", Alphabet: KeyboardASCIIAlphabet,
		Wildcard: KeyboardASCIIWildcard, Scale: 2, MatchMargin: 4,
		GapScore: -1,
	}
}

// ConfusionCount records how often Typed appeared when Intended was expected.
type ConfusionCount struct {
	Typed    rune
	Intended rune
	Count    float64
}

func keyboardOptions(options *KeyboardOptions) KeyboardOptions {
	resolved := DefaultKeyboardOptions()
	if options != nil {
		resolved = *options
	}
	return resolved
}

func keyboardFromGrid(counts [][]float64, options *KeyboardOptions) (*SubstitutionMatrix, error) {
	resolved := keyboardOptions(options)
	symbols := []rune(resolved.Alphabet)
	size := len(symbols)
	if len(counts) != size {
		return nil, fmt.Errorf("stridealign: keyboard count rows do not match alphabet")
	}
	total := 0.0
	rowTotals := make([]float64, size)
	columnTotals := make([]float64, size)
	for row, values := range counts {
		if len(values) != size {
			return nil, fmt.Errorf("stridealign: keyboard count columns do not match alphabet")
		}
		for column, value := range values {
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
				return nil, fmt.Errorf("stridealign: keyboard counts must be finite and non-negative")
			}
			total += value
			rowTotals[row] += value
			columnTotals[column] += value
		}
	}
	scores := make([][]int, size)
	for row := range scores {
		scores[row] = make([]int, size)
	}
	if total == 0 {
		for row := range scores {
			for column := range scores[row] {
				scores[row][column] = -1
			}
			scores[row][row] = 1
		}
	} else {
		computed := make([][]float64, size)
		floor := math.Inf(1)
		for row := range computed {
			computed[row] = make([]float64, size)
			for column := range computed[row] {
				denominator := rowTotals[row] * columnTotals[column]
				value := math.Inf(-1)
				if counts[row][column] > 0 && denominator > 0 {
					value = resolved.Scale * math.Log2(counts[row][column]*total/denominator)
					if value < floor {
						floor = value
					}
				}
				computed[row][column] = value
			}
		}
		if resolved.Floor != nil {
			floor = *resolved.Floor
		} else if math.IsInf(floor, 1) {
			floor = -1
		} else {
			floor = math.Floor(floor)
		}
		bestSubstitution := -128
		for row := range computed {
			for column, value := range computed[row] {
				if math.IsInf(value, 0) || math.IsNaN(value) {
					value = floor
				}
				score := int(math.Round(value))
				if score < -128 {
					score = -128
				} else if score > 127 {
					score = 127
				}
				scores[row][column] = score
				if row != column && score > bestSubstitution {
					bestSubstitution = score
				}
			}
		}
		match := int(math.Round(math.Max(float64(bestSubstitution), 0) + resolved.MatchMargin))
		if match > 127 {
			match = 127
		}
		for index := range scores {
			scores[index][index] = match
		}
	}
	matrixOptions := MatrixOptions{
		Wildcard: resolved.Wildcard, GapScore: resolved.GapScore,
		GapOpenScore:   resolved.GapOpenScore,
		GapExtendScore: resolved.GapExtendScore, AffineGaps: resolved.AffineGaps,
	}
	return NewSubstitutionMatrix(resolved.Name, resolved.Alphabet, scores, &matrixOptions)
}

// KeyboardFromConfusionGrid builds a matrix from a dense typed-by-intended grid.
func KeyboardFromConfusionGrid(counts [][]float64, options *KeyboardOptions) (*SubstitutionMatrix, error) {
	return keyboardFromGrid(counts, options)
}

// KeyboardFromConfusionCounts builds a matrix from sparse observations.
func KeyboardFromConfusionCounts(counts []ConfusionCount, options *KeyboardOptions) (*SubstitutionMatrix, error) {
	resolved := keyboardOptions(options)
	symbols := []rune(resolved.Alphabet)
	indices := make(map[rune]int, len(symbols))
	for index, symbol := range symbols {
		indices[symbol] = index
	}
	grid := make([][]float64, len(symbols))
	for index := range grid {
		grid[index] = make([]float64, len(symbols))
	}
	for _, count := range counts {
		if math.IsNaN(count.Count) || math.IsInf(count.Count, 0) || count.Count < 0 {
			return nil, fmt.Errorf("stridealign: keyboard counts must be finite and non-negative")
		}
		typed, typedOK := indices[count.Typed]
		intended, intendedOK := indices[count.Intended]
		if typedOK && intendedOK {
			grid[typed][intended] += count.Count
		}
	}
	return keyboardFromGrid(grid, &resolved)
}

var (
	npyDescrPattern   = regexp.MustCompile(`['"]descr['"]\s*:\s*['"]\|i1['"]`)
	npyFortranPattern = regexp.MustCompile(`['"]fortran_order['"]\s*:\s*True`)
	npyShapePattern   = regexp.MustCompile(`['"]shape['"]\s*:\s*\(([^)]*)\)`)
)

func readInt8NPY(path string) ([][]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 10 || string(data[:6]) != "\x93NUMPY" {
		return nil, fmt.Errorf("stridealign: not a NumPy .npy file")
	}
	major := data[6]
	headerStart := 10
	headerSize := 0
	if major == 1 {
		headerSize = int(binary.LittleEndian.Uint16(data[8:10]))
	} else if major == 2 || major == 3 {
		if len(data) < 12 {
			return nil, fmt.Errorf("stridealign: truncated NumPy header")
		}
		headerStart = 12
		headerSize = int(binary.LittleEndian.Uint32(data[8:12]))
	} else {
		return nil, fmt.Errorf("stridealign: unsupported NumPy version %d", major)
	}
	if headerSize < 0 || headerStart+headerSize > len(data) {
		return nil, fmt.Errorf("stridealign: truncated NumPy header")
	}
	header := string(data[headerStart : headerStart+headerSize])
	if !npyDescrPattern.MatchString(header) {
		return nil, fmt.Errorf("stridealign: keyboard .npy matrix must use signed int8 values")
	}
	if npyFortranPattern.MatchString(header) {
		return nil, fmt.Errorf("stridealign: Fortran-order .npy matrices are unsupported")
	}
	match := npyShapePattern.FindStringSubmatch(header)
	if len(match) != 2 {
		return nil, fmt.Errorf("stridealign: cannot read NumPy shape")
	}
	var dimensions []int
	for _, field := range strings.Split(match[1], ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		value, parseErr := strconv.Atoi(field)
		if parseErr != nil || value < 0 {
			return nil, fmt.Errorf("stridealign: invalid NumPy shape")
		}
		dimensions = append(dimensions, value)
	}
	if len(dimensions) != 2 || dimensions[0] != dimensions[1] {
		return nil, fmt.Errorf("stridealign: keyboard .npy value must be square and two-dimensional")
	}
	payload := data[headerStart+headerSize:]
	if dimensions[0] != 0 && dimensions[1] > int(^uint(0)>>1)/dimensions[0] {
		return nil, fmt.Errorf("stridealign: NumPy shape is too large")
	}
	count := dimensions[0] * dimensions[1]
	if len(payload) != count {
		return nil, fmt.Errorf("stridealign: NumPy payload size does not match shape")
	}
	grid := make([][]int, dimensions[0])
	for row := range grid {
		grid[row] = make([]int, dimensions[1])
		for column := range grid[row] {
			grid[row][column] = int(int8(payload[row*dimensions[1]+column]))
		}
	}
	return grid, nil
}

// KeyboardFromNPY loads the signed-int8 C-order format used by Python's keyboard matrices.
func KeyboardFromNPY(path string, transpose bool, options *KeyboardOptions) (*SubstitutionMatrix, error) {
	grid, err := readInt8NPY(path)
	if err != nil {
		return nil, err
	}
	resolved := keyboardOptions(options)
	if options == nil {
		resolved.Name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	size := utf8.RuneCountInString(resolved.Alphabet)
	if len(grid) != size {
		return nil, fmt.Errorf("stridealign: keyboard matrix shape does not match alphabet")
	}
	if transpose {
		for row := 0; row < size; row++ {
			for column := row + 1; column < size; column++ {
				grid[row][column], grid[column][row] = grid[column][row], grid[row][column]
			}
		}
	}
	matrixOptions := MatrixOptions{
		Wildcard: resolved.Wildcard, GapScore: resolved.GapScore,
		GapOpenScore:   resolved.GapOpenScore,
		GapExtendScore: resolved.GapExtendScore, AffineGaps: resolved.AffineGaps,
	}
	return NewSubstitutionMatrix(resolved.Name, resolved.Alphabet, grid, &matrixOptions)
}

// KeyboardAvailable lists embedded keyboard matrices.
func KeyboardAvailable() ([]string, error) {
	all, err := AvailableMatrices()
	if err != nil {
		return nil, err
	}
	var names []string
	for _, name := range all {
		if strings.HasPrefix(name, "keyboard:") {
			names = append(names, strings.TrimPrefix(name, "keyboard:"))
		}
	}
	sort.Strings(names)
	return names, nil
}

// KeyboardMatrix loads one embedded keyboard matrix by short name.
func KeyboardMatrix(name string) (*SubstitutionMatrix, error) {
	return BuiltinMatrix("keyboard:" + strings.TrimPrefix(name, "keyboard:"))
}
