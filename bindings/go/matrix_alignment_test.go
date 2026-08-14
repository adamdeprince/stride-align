package stridealign

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestAlignmentPaths(t *testing.T) {
	nw, err := NeedlemanWunschPath("ACGT", "ACCT", nil)
	if err != nil {
		t.Fatal(err)
	}
	if nw.Score != 5 || nw.QueryStart != 0 || nw.QueryEnd != 4 ||
		nw.TargetStart != 0 || nw.TargetEnd != 4 || nw.Operations != "==X=" ||
		nw.CIGAR != "2=1X1=" || nw.AlignedQuery != "ACGT" || nw.AlignedTarget != "ACCT" {
		t.Fatalf("NeedlemanWunschPath = %#v", nw)
	}
	sw, err := SmithWatermanPath("ACCGT", "CCG", nil)
	if err != nil {
		t.Fatal(err)
	}
	if sw.Score != 6 || sw.QueryStart != 1 || sw.QueryEnd != 4 ||
		sw.Operations != "===" || sw.CIGAR != "3=" ||
		sw.AlignedQuery != "CCG" || sw.AlignedTarget != "CCG" {
		t.Fatalf("SmithWatermanPath = %#v", sw)
	}
	options := DefaultScoreOptions()
	options.GapOpenScore = -3
	options.GapExtendScore = -1
	affine, err := SmithWatermanPath("AAABBB", "AAACCCBBB", &options)
	if err != nil || affine.Score != 7 || affine.CIGAR != "3=3I3=" {
		t.Fatalf("affine path = %#v, %v", affine, err)
	}
	unicodePath, err := NeedlemanWunschPath("A🙂", "A🙂", nil)
	if err != nil || unicodePath.Score != 4 || unicodePath.AlignedQuery != "A🙂" {
		t.Fatalf("Unicode path = %#v, %v", unicodePath, err)
	}
	invalid := NewAlignmentScores("ACGT", AlignmentVariant(99), nil)
	if _, err := invalid.Compare([]string{"ACGT"}); err == nil {
		t.Fatal("unknown cached alignment variant was accepted")
	}
}

func TestBuiltinAndCustomMatrices(t *testing.T) {
	names, err := AvailableMatrices()
	if err != nil || !slices.Contains(names, "blosum62") || !slices.Contains(names, "nuc44") {
		t.Fatalf("AvailableMatrices = %#v, %v", names, err)
	}
	blosum, err := BLOSUM62()
	if err != nil {
		t.Fatal(err)
	}
	score, err := blosum.SmithWatermanScore("HE", "HE", nil)
	if err != nil || score != 13 {
		t.Fatalf("BLOSUM62 HE/HE = %d, %v", score, err)
	}
	score, err = blosum.Score("HE", "HH", nil)
	if err != nil || score != 8 {
		t.Fatalf("BLOSUM62 HE/HH = %d, %v", score, err)
	}
	encoded, err := blosum.Encode("H?")
	if err != nil || len(encoded) != 2 || encoded[1] != uint16(blosum.WildcardIndex) {
		t.Fatalf("matrix Encode = %#v, %v", encoded, err)
	}
	limit, err := blosum.ScoreStepLimit(nil)
	if err != nil || limit <= 0 {
		t.Fatalf("ScoreStepLimit = %d, %v", limit, err)
	}

	custom, err := NewSubstitutionMatrix(
		"tiny", "ABX",
		[][]int{{3, -2, -2}, {-1, 4, -2}, {-2, -2, -2}},
		&MatrixOptions{Wildcard: 'X', GapScore: -3},
	)
	if err != nil {
		t.Fatal(err)
	}
	transposed, err := custom.Transpose("")
	if err != nil || transposed.Name != "tiny.T" || transposed.Values[1] != -1 || transposed.Values[3] != -2 {
		t.Fatalf("Transpose = %#v, %v", transposed, err)
	}
}

func TestNCBIIdentityASCIIAndKeyboardMatrices(t *testing.T) {
	ncbi := `# tiny matrix
  A C X
A 3 -1 -2
C -1 3 -2
X -2 -2 -2
`
	parsed, err := SubstitutionMatrixFromNCBIText(ncbi, &NCBIMatrixOptions{
		Name: "TINY", Wildcard: 'X',
		GapOptions: MatrixOptions{GapScore: -2},
	})
	if err != nil || parsed.Name != "TINY" || parsed.Alphabet != "ACX" {
		t.Fatalf("NCBI parse = %#v, %v", parsed, err)
	}
	identity, err := IdentityMatrix("DNA", "ACGT", 5, -4, 'N', nil)
	if err != nil || identity.Alphabet != "ACGTN" {
		t.Fatalf("IdentityMatrix = %#v, %v", identity, err)
	}
	ascii, err := ASCIIMatrix("ASCII", 1, -1, nil)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := ascii.Encode("\x00A\x7f")
	if err != nil || !slices.Equal(encoded, []uint16{0, 65, 127}) {
		t.Fatalf("ASCII Encode = %#v, %v", encoded, err)
	}
	keyboard, err := KeyboardFromConfusionCounts(
		[]ConfusionCount{{Typed: 'a', Intended: 'b', Count: 8}, {Typed: 'a', Intended: 'a', Count: 2}},
		&KeyboardOptions{
			Name: "keys", Alphabet: "ab?", Wildcard: '?', Scale: 2,
			MatchMargin: 4, GapScore: -1,
		},
	)
	if err != nil || keyboard.Name != "keys" || len(keyboard.Values) != 9 {
		t.Fatalf("KeyboardFromConfusionCounts = %#v, %v", keyboard, err)
	}
}

func TestKeyboardNPYLoader(t *testing.T) {
	header := []byte("{'descr' : '|i1', 'fortran_order' : False, 'shape' : (3, 3), }\n")
	contents := append([]byte("\x93NUMPY\x01\x00"), byte(len(header)), byte(len(header)>>8))
	contents = append(contents, header...)
	contents = append(contents, []byte{1, 2, 3, 4, 5, 6, 7, 8, 9}...)
	path := filepath.Join(t.TempDir(), "keyboard.npy")
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	matrix, err := KeyboardFromNPY(path, true, &KeyboardOptions{
		Name: "npy", Alphabet: "ab?", Wildcard: '?', GapScore: -1,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []int8{1, 4, 7, 2, 5, 8, 3, 6, 9}
	if !slices.Equal(matrix.Values, want) {
		t.Fatalf("transposed NumPy values = %#v, want %#v", matrix.Values, want)
	}

	versionTwo := append([]byte("\x93NUMPY\x02\x00"), make([]byte, 4)...)
	binary.LittleEndian.PutUint32(versionTwo[8:12], uint32(len(header)))
	versionTwo = append(versionTwo, header...)
	versionTwo = append(versionTwo, contents[len(contents)-9:]...)
	path = filepath.Join(t.TempDir(), "keyboard-v2.npy")
	if err := os.WriteFile(path, versionTwo, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := KeyboardFromNPY(path, false, &KeyboardOptions{
		Name: "npy-v2", Alphabet: "ab?", Wildcard: '?', GapScore: -1,
	}); err != nil {
		t.Fatalf("NumPy v2 loader: %v", err)
	}
}

func TestMatrixCDistForms(t *testing.T) {
	matrix, err := BLOSUM62()
	if err != nil {
		t.Fatal(err)
	}
	queries := []string{"HE", "HH"}
	targets := []string{"HE", "HH"}
	dense, err := matrix.CDist(queries, targets, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float64{{13, 8}, {8, 16}}
	for row := range want {
		if !slices.Equal(dense[row], want[row]) {
			t.Fatalf("matrix CDist = %#v, want %#v", dense, want)
		}
	}
	filtered, err := matrix.CDistAboveThreshold(queries, targets, true, 13, nil)
	if err != nil || len(filtered) != 2 {
		t.Fatalf("matrix threshold = %#v, %v", filtered, err)
	}
	top, err := matrix.CDistTopK(queries, targets, true, 2, nil)
	if err != nil || len(top) != 2 || top[0].Score != 16 || top[1].Score != 13 {
		t.Fatalf("matrix top-k = %#v, %v", top, err)
	}
}
