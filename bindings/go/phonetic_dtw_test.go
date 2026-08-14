package stridealign

import "testing"

func TestPhoneticCatalog(t *testing.T) {
	stringCases := []struct {
		name string
		call func() (string, error)
		want string
	}{
		{"soundex", func() (string, error) { return Soundex("Ashcraft") }, "A261"},
		{"metaphone", func() (string, error) { return Metaphone("Thompson", MetaphonePhilips) }, "0MPSN"},
		{"nysiis", func() (string, error) { return NYSIIS("Watkins") }, "WATCAN"},
		{"match-rating", func() (string, error) { return MatchRatingCodex("Catherine") }, "CTHRN"},
		{"caverphone", func() (string, error) { return Caverphone("Stevenson") }, "STFNSN1111"},
		{"cologne", func() (string, error) { return ColognePhonetic("Müller") }, "657"},
		{"daitch-mokotoff", func() (string, error) { return DaitchMokotoff("Washington", true, true) }, "746536"},
	}
	for _, test := range stringCases {
		got, err := test.call()
		if err != nil || got != test.want {
			t.Errorf("%s = %q, %v; want %q", test.name, got, err, test.want)
		}
	}
	for name, call := range map[string]func() (bool, error){
		"soundex":      func() (bool, error) { return SoundexEqual("Robert", "Rupert") },
		"metaphone":    func() (bool, error) { return MetaphoneEqual("Catherine", "Kathryn", MetaphonePhilips) },
		"nysiis":       func() (bool, error) { return NYSIISEqual("Catherine", "Catharine") },
		"match-rating": func() (bool, error) { return MatchRatingCompare("Robert", "Rupert") },
	} {
		got, err := call()
		if err != nil || !got {
			t.Errorf("%s equality = %v, %v", name, got, err)
		}
	}
	if equal, err := SoundexEqual("", ""); err != nil || equal {
		t.Fatalf("empty SoundexEqual = %v, %v", equal, err)
	}
}

func TestDoubleMetaphoneAndBeiderMorse(t *testing.T) {
	double, err := DoubleMetaphone("Smith", 64, DoubleMetaphoneCommons)
	if err != nil || double.Primary != "SM0" || double.Alternate != "XMT" {
		t.Fatalf("DoubleMetaphone = %#v, %v", double, err)
	}
	short, err := DoubleMetaphone("Christopher", 4, DoubleMetaphoneCommons)
	if err != nil || len(short.Primary) > 4 || len(short.Alternate) > 4 {
		t.Fatalf("short DoubleMetaphone = %#v, %v", short, err)
	}
	doubles, err := DoubleMetaphoneAll(
		[]string{"Smith", "Christopher"}, 4, DoubleMetaphoneCommons,
	)
	if err != nil || len(doubles) != 2 || doubles[0].Primary != "SM0" || len(doubles[1].Primary) > 4 {
		t.Fatalf("DoubleMetaphoneAll = %#v, %v", doubles, err)
	}
	bmpm, err := BeiderMorse("Renault", BmpmApprox, true, 10)
	if err != nil || bmpm != "rinD|rinDlt|rina|rinalt|rino|rinolt|rinu|rinult" {
		t.Fatalf("BeiderMorse = %q, %v", bmpm, err)
	}
	all, err := SoundexAll([]string{"Robert", "Rupert", "Rubin"})
	if err != nil || len(all) != 3 || all[0] != "R163" || all[1] != "R163" || all[2] != "R150" {
		t.Fatalf("SoundexAll = %#v, %v", all, err)
	}
	equals, err := SoundexEquals("Robert", []string{"Rupert", "Smith", ""})
	if err != nil || len(equals) != 3 || !equals[0] || equals[1] || equals[2] {
		t.Fatalf("SoundexEquals = %#v, %v", equals, err)
	}
}

func TestDTWAndBatch(t *testing.T) {
	distance, err := DTW([]float64{1, 2, 3}, []float64{2, 2, 4}, nil)
	if err != nil || distance != 2 {
		t.Fatalf("DTW = %v, %v", distance, err)
	}
	options := DefaultDTWOptions()
	options.Distance = DTWL1
	distance, err = DTW([]float64{1, 2, 3}, []float64{2, 2, 4}, &options)
	if err != nil || distance != 2 {
		t.Fatalf("L1 DTW = %v, %v", distance, err)
	}
	distances, err := DTWDistances(
		[]float64{1, 2, 3},
		[][]float64{{1, 2, 3}, {2, 2, 4}},
		nil,
	)
	if err != nil || len(distances) != 2 || distances[0] != 0 || distances[1] != 2 {
		t.Fatalf("DTWDistances = %#v, %v", distances, err)
	}
	integer, err := DTWInts([]int64{1, 2, 3}, []int64{2, 2, 4}, nil)
	if err != nil || integer != 2 {
		t.Fatalf("DTWInts = %v, %v", integer, err)
	}
	integerDistances, err := DTWIntsDistances(
		[]int64{1, 2, 3}, [][]int64{{1, 2, 3}, {2, 2, 4}}, nil,
	)
	if err != nil || len(integerDistances) != 2 || integerDistances[0] != 0 || integerDistances[1] != 2 {
		t.Fatalf("DTWIntsDistances = %#v, %v", integerDistances, err)
	}
	if _, err := DTW(nil, []float64{1}, nil); err == nil {
		t.Fatal("empty DTW input accepted")
	}
}
