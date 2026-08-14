package stridealign

const (
	unarySoundex = iota
	unaryMetaphone
	unaryNYSIIS
	unaryMatchRatingCodex
	unaryCaverphone
	unaryColognePhonetic
	unaryDaitchMokotoff
	unaryBeiderMorse
)

const (
	pairSoundexEqual = iota
	pairMetaphoneEqual
	pairNYSIISEqual
	pairMatchRatingCompare
)

// MetaphoneVariant selects the Philips or Jellyfish-compatible rule details.
type MetaphoneVariant int

const (
	MetaphonePhilips MetaphoneVariant = iota
	MetaphoneJellyfish
)

// DoubleMetaphoneVariant selects Apache Commons or Python-compatible details.
type DoubleMetaphoneVariant int

const (
	DoubleMetaphoneCommons DoubleMetaphoneVariant = iota
	DoubleMetaphonePython
)

// BmpmRuleType selects approximate or exact Beider-Morse rules.
type BmpmRuleType int

const (
	BmpmApprox BmpmRuleType = iota
	BmpmExact
)

// DoubleMetaphoneResult contains the primary and alternate encodings.
type DoubleMetaphoneResult struct {
	Primary   string
	Alternate string
}

func Soundex(input string) (string, error) {
	return nativeUnaryString(unarySoundex, input, 0, 0, false, false)
}

func SoundexEqual(query, target string) (bool, error) {
	return nativePairBool(pairSoundexEqual, query, target, 0)
}

func Metaphone(input string, variant MetaphoneVariant) (string, error) {
	return nativeUnaryString(unaryMetaphone, input, int(variant), 0, false, false)
}

func MetaphoneEqual(query, target string, variant MetaphoneVariant) (bool, error) {
	return nativePairBool(pairMetaphoneEqual, query, target, int(variant))
}

func NYSIIS(input string) (string, error) {
	return nativeUnaryString(unaryNYSIIS, input, 0, 0, false, false)
}

func NYSIISEqual(query, target string) (bool, error) {
	return nativePairBool(pairNYSIISEqual, query, target, 0)
}

func MatchRatingCodex(input string) (string, error) {
	return nativeUnaryString(unaryMatchRatingCodex, input, 0, 0, false, false)
}

func MatchRatingCompare(query, target string) (bool, error) {
	return nativePairBool(pairMatchRatingCompare, query, target, 0)
}

func Caverphone(input string) (string, error) {
	return nativeUnaryString(unaryCaverphone, input, 0, 0, false, false)
}

func ColognePhonetic(input string) (string, error) {
	return nativeUnaryString(unaryColognePhonetic, input, 0, 0, false, false)
}

// DaitchMokotoff returns one code or a pipe-delimited set of codes when
// branching is enabled.
func DaitchMokotoff(input string, branching, folding bool) (string, error) {
	return nativeUnaryString(unaryDaitchMokotoff, input, 0, 0, branching, folding)
}

func DoubleMetaphone(input string, maximumLength int, variant DoubleMetaphoneVariant) (DoubleMetaphoneResult, error) {
	return nativeDoubleMetaphone(input, maximumLength, int(variant))
}

func DoubleMetaphoneAll(inputs []string, maximumLength int, variant DoubleMetaphoneVariant) ([]DoubleMetaphoneResult, error) {
	return nativeDoubleMetaphones(inputs, maximumLength, int(variant))
}

// BeiderMorse applies embedded BMPM resources; callers do not need data files.
func BeiderMorse(input string, ruleType BmpmRuleType, concat bool, maxPhonemes int) (string, error) {
	return nativeUnaryString(
		unaryBeiderMorse, input, int(ruleType), maxPhonemes, concat, false,
	)
}

func unaryStrings(inputs []string, operation, optionA, optionB int, boolA, boolB bool) ([]string, error) {
	return nativeUnaryStrings(operation, inputs, optionA, optionB, boolA, boolB)
}

// SoundexAll encodes a slice while preserving input order.
func SoundexAll(inputs []string) ([]string, error) {
	return unaryStrings(inputs, unarySoundex, 0, 0, false, false)
}

func MetaphoneAll(inputs []string, variant MetaphoneVariant) ([]string, error) {
	return unaryStrings(inputs, unaryMetaphone, int(variant), 0, false, false)
}

func NYSIISAll(inputs []string) ([]string, error) {
	return unaryStrings(inputs, unaryNYSIIS, 0, 0, false, false)
}

func MatchRatingCodexAll(inputs []string) ([]string, error) {
	return unaryStrings(inputs, unaryMatchRatingCodex, 0, 0, false, false)
}

func CaverphoneAll(inputs []string) ([]string, error) {
	return unaryStrings(inputs, unaryCaverphone, 0, 0, false, false)
}

func ColognePhoneticAll(inputs []string) ([]string, error) {
	return unaryStrings(inputs, unaryColognePhonetic, 0, 0, false, false)
}

func DaitchMokotoffAll(inputs []string, branching, folding bool) ([]string, error) {
	return unaryStrings(inputs, unaryDaitchMokotoff, 0, 0, branching, folding)
}

func BeiderMorseAll(inputs []string, ruleType BmpmRuleType, concat bool, maxPhonemes int) ([]string, error) {
	return unaryStrings(inputs, unaryBeiderMorse, int(ruleType), maxPhonemes, concat, false)
}

func SoundexEquals(query string, targets []string) ([]bool, error) {
	return nativePairBools(pairSoundexEqual, query, targets, 0)
}

func MetaphoneEquals(query string, targets []string, variant MetaphoneVariant) ([]bool, error) {
	return nativePairBools(pairMetaphoneEqual, query, targets, int(variant))
}

func NYSIISEquals(query string, targets []string) ([]bool, error) {
	return nativePairBools(pairNYSIISEqual, query, targets, 0)
}

func MatchRatingCompares(query string, targets []string) ([]bool, error) {
	return nativePairBools(pairMatchRatingCompare, query, targets, 0)
}
