package stridealign

/*
#cgo !stridealign_prebuilt CXXFLAGS: -std=c++23 -O3 -DNDEBUG -I${SRCDIR}/../../include
#cgo linux,!stridealign_prebuilt LDFLAGS: -lstdc++ -lm
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

func nativeVersion() string { return C.GoString(C.stride_go_version()) }

func nativeSIMDLevel() string { return C.GoString(C.stride_go_simd_level()) }

type nativeFailure struct {
	operation string
	message   string
}

func (failure *nativeFailure) Error() string {
	return fmt.Sprintf("stridealign: %s: %s", failure.operation, failure.message)
}

func nativeString(value string) C.stride_string {
	var data *C.char
	if len(value) != 0 {
		data = (*C.char)(unsafe.Pointer(unsafe.StringData(value)))
	}
	return C.stride_string{data: data, size: C.size_t(len(value))}
}

type packedStrings struct {
	data    []byte
	offsets []C.size_t
	view    C.stride_strings
}

func packStrings(values []string) packedStrings {
	total := 0
	for _, value := range values {
		if len(value) > int(^uint(0)>>1)-total {
			panic("stridealign: string collection is too large")
		}
		total += len(value)
	}
	packed := packedStrings{
		data:    make([]byte, 0, total),
		offsets: make([]C.size_t, len(values)+1),
	}
	for index, value := range values {
		packed.data = append(packed.data, value...)
		packed.offsets[index+1] = C.size_t(len(packed.data))
	}
	if len(packed.data) != 0 {
		packed.view.data = (*C.char)(unsafe.Pointer(&packed.data[0]))
	}
	packed.view.data_size = C.size_t(len(packed.data))
	if len(packed.offsets) != 0 {
		packed.view.offsets = (*C.size_t)(unsafe.Pointer(&packed.offsets[0]))
	}
	packed.view.count = C.size_t(len(values))
	return packed
}

func (packed *packedStrings) keepAlive() {
	runtime.KeepAlive(packed.data)
	runtime.KeepAlive(packed.offsets)
}

func checkedLength(size C.size_t) (int, error) {
	value := uint64(size)
	maximum := uint64(^uint(0) >> 1)
	if value > maximum {
		return 0, fmt.Errorf("stridealign: native result is too large")
	}
	return int(value), nil
}

func copyBytes(data *C.char, size C.size_t) (string, error) {
	length, err := checkedLength(size)
	if err != nil {
		return "", err
	}
	if length == 0 {
		return "", nil
	}
	if data == nil {
		return "", fmt.Errorf("stridealign: native result has null data")
	}
	source := unsafe.Slice((*byte)(unsafe.Pointer(data)), length)
	return string(source), nil
}

func takeBytes(value *C.stride_bytes) (string, error) {
	defer C.stride_go_free(unsafe.Pointer(value.data))
	return copyBytes(value.data, value.size)
}

func nativeError(operation string, status C.int, failure *C.stride_bytes) error {
	if status == 0 {
		if failure.data != nil {
			C.stride_go_free(unsafe.Pointer(failure.data))
			*failure = C.stride_bytes{}
		}
		return nil
	}
	message, err := takeBytes(failure)
	if err != nil || message == "" {
		message = "native operation failed without an error message"
	}
	return &nativeFailure{operation: operation, message: message}
}

func cScoreOptions(options *ScoreOptions) (C.stride_score_options, error) {
	resolved := DefaultScoreOptions()
	if options != nil {
		resolved = *options
	}
	if resolved.PrefixCap < 0 {
		return C.stride_score_options{}, fmt.Errorf("stridealign: prefix cap must be non-negative")
	}
	return C.stride_score_options{
		match_score:      C.int64_t(resolved.MatchScore),
		mismatch_score:   C.int64_t(resolved.MismatchScore),
		gap_open_score:   C.int64_t(resolved.GapOpenScore),
		gap_extend_score: C.int64_t(resolved.GapExtendScore),
		prefix_weight:    C.double(resolved.PrefixWeight),
		prefix_threshold: C.double(resolved.PrefixThreshold),
		prefix_cap:       C.size_t(resolved.PrefixCap),
	}, nil
}

func nativeScore(
	query string,
	target string,
	scorer Scorer,
	options *ScoreOptions,
) (float64, error) {
	resolved, err := cScoreOptions(options)
	if err != nil {
		return 0, err
	}
	var output C.double
	var failure C.stride_bytes
	status := C.stride_go_score(
		C.int(scorer),
		nativeString(query),
		nativeString(target),
		resolved,
		&output,
		&failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	if err := nativeError("score", status, &failure); err != nil {
		return 0, err
	}
	return float64(output), nil
}

func takeDoubles(output *C.stride_doubles) ([]float64, error) {
	defer C.stride_go_free(unsafe.Pointer(output.data))
	length, err := checkedLength(output.size)
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return []float64{}, nil
	}
	if output.data == nil {
		return nil, fmt.Errorf("stridealign: native double result has null data")
	}
	source := unsafe.Slice((*float64)(unsafe.Pointer(output.data)), length)
	return append([]float64(nil), source...), nil
}

func nativeScores(
	query string,
	targets []string,
	scorer Scorer,
	options *ScoreOptions,
) ([]float64, error) {
	resolved, err := cScoreOptions(options)
	if err != nil {
		return nil, err
	}
	packed := packStrings(targets)
	var output C.stride_doubles
	var failure C.stride_bytes
	status := C.stride_go_scores(
		C.int(scorer), nativeString(query), packed.view,
		resolved, &output, &failure,
	)
	runtime.KeepAlive(query)
	packed.keepAlive()
	if err := nativeError("scores", status, &failure); err != nil {
		return nil, err
	}
	return takeDoubles(&output)
}

func nativeTopK(
	query string,
	targets []string,
	scorer Scorer,
	k int,
	options *ScoreOptions,
	skipInvalidHamming bool,
) ([]Match, error) {
	if k < 0 {
		return nil, fmt.Errorf("stridealign: k must be non-negative")
	}
	resolved, err := cScoreOptions(options)
	if err != nil {
		return nil, err
	}
	packed := packStrings(targets)
	var output C.stride_ranked_matches
	var failure C.stride_bytes
	status := C.stride_go_top_k(
		C.int(scorer), nativeString(query), packed.view, C.size_t(k),
		resolved, C.int(boolInt(skipInvalidHamming)),
		&output, &failure,
	)
	runtime.KeepAlive(query)
	packed.keepAlive()
	if err := nativeError("top_k", status, &failure); err != nil {
		return nil, err
	}
	defer C.stride_go_free(unsafe.Pointer(output.data))
	length, err := checkedLength(output.size)
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return []Match{}, nil
	}
	if output.data == nil {
		return nil, fmt.Errorf("stridealign: native top-k result has null data")
	}
	values := unsafe.Slice((*C.stride_ranked_match)(unsafe.Pointer(output.data)), length)
	matches := make([]Match, length)
	for index, value := range values {
		targetIndex := int(value.index)
		if targetIndex < 0 || targetIndex >= len(targets) {
			return nil, fmt.Errorf("stridealign: native top-k index is out of range")
		}
		matches[index] = Match{
			Target: targets[targetIndex],
			Score:  float64(value.score),
			Index:  targetIndex,
		}
	}
	return matches, nil
}

func nativeCDist(
	queries []string,
	targets []string,
	scorer Scorer,
	options *ScoreOptions,
) ([][]float64, error) {
	resolved, err := cScoreOptions(options)
	if err != nil {
		return nil, err
	}
	packedQueries := packStrings(queries)
	packedTargets := packStrings(targets)
	var output C.stride_doubles
	var failure C.stride_bytes
	status := C.stride_go_cdist(
		C.int(scorer), packedQueries.view, packedTargets.view,
		resolved, &output, &failure,
	)
	packedQueries.keepAlive()
	packedTargets.keepAlive()
	if err := nativeError("cdist", status, &failure); err != nil {
		return nil, err
	}
	flattened, err := takeDoubles(&output)
	if err != nil {
		return nil, err
	}
	if len(queries) != 0 && len(targets) > int(^uint(0)>>1)/len(queries) {
		return nil, fmt.Errorf("stridealign: cdist dimensions overflow")
	}
	if len(flattened) != len(queries)*len(targets) {
		return nil, fmt.Errorf("stridealign: native cdist shape is inconsistent")
	}
	rows := make([][]float64, len(queries))
	for index := range rows {
		begin := index * len(targets)
		rows[index] = flattened[begin : begin+len(targets)]
	}
	return rows, nil
}

func takeMatrixMatches(
	output *C.stride_matrix_matches,
	queries []string,
	targets []string,
) ([]PairMatch, error) {
	defer C.stride_go_free(unsafe.Pointer(output.data))
	length, err := checkedLength(output.size)
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return []PairMatch{}, nil
	}
	if output.data == nil {
		return nil, fmt.Errorf("stridealign: native pair-match result has null data")
	}
	values := unsafe.Slice((*C.stride_matrix_match)(unsafe.Pointer(output.data)), length)
	matches := make([]PairMatch, length)
	for index, value := range values {
		queryIndex := int(value.query_index)
		targetIndex := int(value.target_index)
		if queryIndex < 0 || queryIndex >= len(queries) ||
			targetIndex < 0 || targetIndex >= len(targets) {
			return nil, fmt.Errorf("stridealign: native pair-match index is out of range")
		}
		matches[index] = PairMatch{
			Query:       queries[queryIndex],
			Target:      targets[targetIndex],
			Score:       float64(value.score),
			QueryIndex:  queryIndex,
			TargetIndex: targetIndex,
		}
	}
	return matches, nil
}

func nativeCDistAboveThreshold(
	queries []string,
	targets []string,
	scorer Scorer,
	threshold float64,
	options *ScoreOptions,
) ([]PairMatch, error) {
	resolved, err := cScoreOptions(options)
	if err != nil {
		return nil, err
	}
	packedQueries := packStrings(queries)
	packedTargets := packStrings(targets)
	var output C.stride_matrix_matches
	var failure C.stride_bytes
	status := C.stride_go_cdist_above_threshold(
		C.int(scorer), packedQueries.view, packedTargets.view,
		C.double(threshold), resolved, &output, &failure,
	)
	packedQueries.keepAlive()
	packedTargets.keepAlive()
	if err := nativeError("cdist_above_threshold", status, &failure); err != nil {
		return nil, err
	}
	return takeMatrixMatches(&output, queries, targets)
}

func nativeCDistTopK(
	queries []string,
	targets []string,
	scorer Scorer,
	k int,
	rejectDuplicates bool,
	options *ScoreOptions,
) ([]PairMatch, error) {
	if k < 0 {
		return nil, fmt.Errorf("stridealign: k must be non-negative")
	}
	resolved, err := cScoreOptions(options)
	if err != nil {
		return nil, err
	}
	packedQueries := packStrings(queries)
	packedTargets := packStrings(targets)
	var output C.stride_matrix_matches
	var failure C.stride_bytes
	status := C.stride_go_cdist_top_k(
		C.int(scorer), packedQueries.view, packedTargets.view, C.size_t(k),
		C.int(boolInt(rejectDuplicates)), resolved,
		&output, &failure,
	)
	packedQueries.keepAlive()
	packedTargets.keepAlive()
	if err := nativeError("cdist_top_k", status, &failure); err != nil {
		return nil, err
	}
	return takeMatrixMatches(&output, queries, targets)
}

func nativeCDistTopKPerQuery(
	queries []string,
	targets []string,
	scorer Scorer,
	k int,
	options *ScoreOptions,
) ([]QueryMatches, error) {
	if k < 0 {
		return nil, fmt.Errorf("stridealign: k must be non-negative")
	}
	resolved, err := cScoreOptions(options)
	if err != nil {
		return nil, err
	}
	packedQueries := packStrings(queries)
	packedTargets := packStrings(targets)
	var output C.stride_grouped_matches
	var failure C.stride_bytes
	status := C.stride_go_cdist_top_k_per_query(
		C.int(scorer), packedQueries.view, packedTargets.view,
		C.size_t(k), resolved, &output, &failure,
	)
	packedQueries.keepAlive()
	packedTargets.keepAlive()
	if err := nativeError("cdist_top_k_per_query", status, &failure); err != nil {
		return nil, err
	}
	defer C.stride_go_free(unsafe.Pointer(output.data))
	defer C.stride_go_free(unsafe.Pointer(output.offsets))
	dataLength, err := checkedLength(output.data_size)
	if err != nil {
		return nil, err
	}
	queryCount, err := checkedLength(output.query_count)
	if err != nil {
		return nil, err
	}
	if queryCount != len(queries) {
		return nil, fmt.Errorf("stridealign: native per-query result shape is inconsistent")
	}
	if output.offsets == nil && queryCount != 0 {
		return nil, fmt.Errorf("stridealign: native per-query offsets are null")
	}
	var values []C.stride_ranked_match
	if dataLength != 0 {
		if output.data == nil {
			return nil, fmt.Errorf("stridealign: native per-query data is null")
		}
		values = unsafe.Slice(
			(*C.stride_ranked_match)(unsafe.Pointer(output.data)), dataLength,
		)
	}
	offsets := unsafe.Slice((*C.size_t)(unsafe.Pointer(output.offsets)), queryCount+1)
	groups := make([]QueryMatches, queryCount)
	for queryIndex := range groups {
		begin := int(offsets[queryIndex])
		end := int(offsets[queryIndex+1])
		if begin < 0 || begin > end || end > len(values) {
			return nil, fmt.Errorf("stridealign: native per-query offsets are invalid")
		}
		matches := make([]Match, end-begin)
		for index, value := range values[begin:end] {
			targetIndex := int(value.index)
			if targetIndex < 0 || targetIndex >= len(targets) {
				return nil, fmt.Errorf("stridealign: native target index is out of range")
			}
			matches[index] = Match{
				Target: targets[targetIndex],
				Score:  float64(value.score),
				Index:  targetIndex,
			}
		}
		groups[queryIndex] = QueryMatches{
			Query:   queries[queryIndex],
			Index:   queryIndex,
			Matches: matches,
		}
	}
	return groups, nil
}

func nativePairNumber(
	operation int,
	query string,
	target string,
	integerOption int,
	stringOption string,
	boolOption bool,
) (float64, error) {
	if integerOption < 0 {
		return 0, fmt.Errorf("stridealign: integer option must be non-negative")
	}
	var output C.double
	var failure C.stride_bytes
	status := C.stride_go_pair_number(
		C.int(operation), nativeString(query), nativeString(target),
		C.size_t(integerOption), nativeString(stringOption),
		C.int(boolInt(boolOption)), &output, &failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	runtime.KeepAlive(stringOption)
	if err := nativeError("pair operation", status, &failure); err != nil {
		return 0, err
	}
	return float64(output), nil
}

func nativePairNumbers(
	operation int,
	query string,
	targets []string,
	integerOption int,
	stringOption string,
	boolOption bool,
) ([]float64, error) {
	if integerOption < 0 {
		return nil, fmt.Errorf("stridealign: integer option must be non-negative")
	}
	packed := packStrings(targets)
	var output C.stride_doubles
	var failure C.stride_bytes
	status := C.stride_go_pair_numbers(
		C.int(operation), nativeString(query), packed.view,
		C.size_t(integerOption), nativeString(stringOption),
		C.int(boolInt(boolOption)), &output, &failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(stringOption)
	packed.keepAlive()
	if err := nativeError("pair operations", status, &failure); err != nil {
		return nil, err
	}
	return takeDoubles(&output)
}

func nativeLCSSubstring(query string, target string) (string, error) {
	var output C.stride_bytes
	var failure C.stride_bytes
	status := C.stride_go_lcs_substring(
		nativeString(query), nativeString(target), &output, &failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	if err := nativeError("lcs_substring", status, &failure); err != nil {
		return "", err
	}
	return takeBytes(&output)
}

func nativeLCSSubstrings(query string, targets []string) ([]string, error) {
	packed := packStrings(targets)
	var output C.stride_string_list
	var failure C.stride_bytes
	status := C.stride_go_lcs_substrings(
		nativeString(query), packed.view, &output, &failure,
	)
	runtime.KeepAlive(query)
	packed.keepAlive()
	if err := nativeError("lcs_substrings", status, &failure); err != nil {
		return nil, err
	}
	return takeStringList(&output)
}

func nativePairBool(
	operation int,
	query string,
	target string,
	variant int,
) (bool, error) {
	var output C.int
	var failure C.stride_bytes
	status := C.stride_go_pair_bool(
		C.int(operation), nativeString(query), nativeString(target),
		C.int(variant), &output, &failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	if err := nativeError("phonetic comparison", status, &failure); err != nil {
		return false, err
	}
	return output != 0, nil
}

func nativePairBools(
	operation int,
	query string,
	targets []string,
	variant int,
) ([]bool, error) {
	packed := packStrings(targets)
	var output C.stride_uints
	var failure C.stride_bytes
	status := C.stride_go_pair_bools(
		C.int(operation), nativeString(query), packed.view, C.int(variant),
		&output, &failure,
	)
	runtime.KeepAlive(query)
	packed.keepAlive()
	if err := nativeError("phonetic comparisons", status, &failure); err != nil {
		return nil, err
	}
	defer C.stride_go_free(unsafe.Pointer(output.data))
	length, err := checkedLength(output.size)
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return []bool{}, nil
	}
	if output.data == nil {
		return nil, fmt.Errorf("stridealign: native boolean result is null")
	}
	values := unsafe.Slice((*uint64)(unsafe.Pointer(output.data)), length)
	result := make([]bool, length)
	for index, value := range values {
		result[index] = value != 0
	}
	return result, nil
}

func nativeUnaryString(
	operation int,
	input string,
	integerOptionA int,
	integerOptionB int,
	boolOptionA bool,
	boolOptionB bool,
) (string, error) {
	var output C.stride_bytes
	var failure C.stride_bytes
	status := C.stride_go_unary_string(
		C.int(operation), nativeString(input), C.int64_t(integerOptionA),
		C.int64_t(integerOptionB), C.int(boolInt(boolOptionA)),
		C.int(boolInt(boolOptionB)), &output, &failure,
	)
	runtime.KeepAlive(input)
	if err := nativeError("phonetic encoding", status, &failure); err != nil {
		return "", err
	}
	return takeBytes(&output)
}

func nativeUnaryStrings(
	operation int,
	inputs []string,
	integerOptionA int,
	integerOptionB int,
	boolOptionA bool,
	boolOptionB bool,
) ([]string, error) {
	packed := packStrings(inputs)
	var output C.stride_string_list
	var failure C.stride_bytes
	status := C.stride_go_unary_strings(
		C.int(operation), packed.view, C.int64_t(integerOptionA),
		C.int64_t(integerOptionB), C.int(boolInt(boolOptionA)),
		C.int(boolInt(boolOptionB)), &output, &failure,
	)
	packed.keepAlive()
	if err := nativeError("phonetic encodings", status, &failure); err != nil {
		return nil, err
	}
	return takeStringList(&output)
}

func nativeDoubleMetaphone(
	input string,
	maximumLength int,
	variant int,
) (DoubleMetaphoneResult, error) {
	if maximumLength < 0 {
		return DoubleMetaphoneResult{}, fmt.Errorf("stridealign: maximum length must be non-negative")
	}
	var primary C.stride_bytes
	var alternate C.stride_bytes
	var failure C.stride_bytes
	status := C.stride_go_double_metaphone(
		nativeString(input), C.size_t(maximumLength), C.int(variant),
		&primary, &alternate, &failure,
	)
	runtime.KeepAlive(input)
	if err := nativeError("double_metaphone", status, &failure); err != nil {
		return DoubleMetaphoneResult{}, err
	}
	primaryValue, err := takeBytes(&primary)
	if err != nil {
		C.stride_go_free(unsafe.Pointer(alternate.data))
		return DoubleMetaphoneResult{}, err
	}
	alternateValue, err := takeBytes(&alternate)
	if err != nil {
		return DoubleMetaphoneResult{}, err
	}
	return DoubleMetaphoneResult{Primary: primaryValue, Alternate: alternateValue}, nil
}

func nativeDoubleMetaphones(
	inputs []string,
	maximumLength int,
	variant int,
) ([]DoubleMetaphoneResult, error) {
	if maximumLength < 0 {
		return nil, fmt.Errorf("stridealign: maximum length must be non-negative")
	}
	packed := packStrings(inputs)
	var primaries C.stride_string_list
	var alternates C.stride_string_list
	var failure C.stride_bytes
	status := C.stride_go_double_metaphones(
		packed.view, C.size_t(maximumLength), C.int(variant),
		&primaries, &alternates, &failure,
	)
	packed.keepAlive()
	if err := nativeError("double_metaphones", status, &failure); err != nil {
		return nil, err
	}
	primaryValues, err := takeStringList(&primaries)
	if err != nil {
		C.stride_go_free(unsafe.Pointer(alternates.data))
		C.stride_go_free(unsafe.Pointer(alternates.offsets))
		return nil, err
	}
	alternateValues, err := takeStringList(&alternates)
	if err != nil {
		return nil, err
	}
	if len(primaryValues) != len(inputs) || len(alternateValues) != len(inputs) {
		return nil, fmt.Errorf("stridealign: native double-metaphone shape is inconsistent")
	}
	output := make([]DoubleMetaphoneResult, len(inputs))
	for index := range output {
		output[index] = DoubleMetaphoneResult{
			Primary: primaryValues[index], Alternate: alternateValues[index],
		}
	}
	return output, nil
}

func nativeDTW(query []float64, target []float64, options *DTWOptions) (float64, error) {
	resolved := DefaultDTWOptions()
	if options != nil {
		resolved = *options
	}
	var queryData *C.double
	if len(query) != 0 {
		queryData = (*C.double)(unsafe.Pointer(&query[0]))
	}
	var targetData *C.double
	if len(target) != 0 {
		targetData = (*C.double)(unsafe.Pointer(&target[0]))
	}
	var output C.double
	var failure C.stride_bytes
	status := C.stride_go_dtw(
		queryData, C.size_t(len(query)), targetData, C.size_t(len(target)),
		C.double(resolved.Window), C.int(resolved.Distance),
		C.double(resolved.ScoreCutoff), &output, &failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	if err := nativeError("dtw", status, &failure); err != nil {
		return 0, err
	}
	return float64(output), nil
}

func nativeDTWDistances(query []float64, targets [][]float64, options *DTWOptions) ([]float64, error) {
	resolved := DefaultDTWOptions()
	if options != nil {
		resolved = *options
	}
	total := 0
	for _, target := range targets {
		if len(target) > int(^uint(0)>>1)-total {
			return nil, fmt.Errorf("stridealign: DTW target collection is too large")
		}
		total += len(target)
	}
	flattened := make([]float64, 0, total)
	offsets := make([]C.size_t, len(targets)+1)
	for index, target := range targets {
		flattened = append(flattened, target...)
		offsets[index+1] = C.size_t(len(flattened))
	}
	var queryData *C.double
	if len(query) != 0 {
		queryData = (*C.double)(unsafe.Pointer(&query[0]))
	}
	var targetData *C.double
	if len(flattened) != 0 {
		targetData = (*C.double)(unsafe.Pointer(&flattened[0]))
	}
	view := C.stride_double_sequences{
		data:      targetData,
		data_size: C.size_t(len(flattened)),
		offsets:   (*C.size_t)(unsafe.Pointer(&offsets[0])),
		count:     C.size_t(len(targets)),
	}
	var output C.stride_doubles
	var failure C.stride_bytes
	status := C.stride_go_dtw_distances(
		queryData, C.size_t(len(query)), view, C.double(resolved.Window),
		C.int(resolved.Distance), C.double(resolved.ScoreCutoff),
		&output, &failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(flattened)
	runtime.KeepAlive(offsets)
	if err := nativeError("dtw_distances", status, &failure); err != nil {
		return nil, err
	}
	return takeDoubles(&output)
}

func nativePath(
	local bool,
	query string,
	target string,
	options *ScoreOptions,
) (AlignmentPath, error) {
	resolved, err := cScoreOptions(options)
	if err != nil {
		return AlignmentPath{}, err
	}
	var output C.stride_alignment_path
	var failure C.stride_bytes
	status := C.stride_go_alignment_path(
		C.int(boolInt(local)), nativeString(query), nativeString(target),
		resolved, &output, &failure,
	)
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	if err := nativeError("alignment path", status, &failure); err != nil {
		return AlignmentPath{}, err
	}
	return takePath(&output)
}

func takePath(output *C.stride_alignment_path) (AlignmentPath, error) {
	defer C.stride_go_free_path(output)
	operations, err := copyBytes(output.operations.data, output.operations.size)
	if err != nil {
		return AlignmentPath{}, err
	}
	cigar, err := copyBytes(output.cigar.data, output.cigar.size)
	if err != nil {
		return AlignmentPath{}, err
	}
	alignedQuery, err := copyBytes(output.aligned_query.data, output.aligned_query.size)
	if err != nil {
		return AlignmentPath{}, err
	}
	alignedTarget, err := copyBytes(output.aligned_target.data, output.aligned_target.size)
	if err != nil {
		return AlignmentPath{}, err
	}
	return AlignmentPath{
		Score:         int64(output.score),
		QueryStart:    int(output.query_start),
		QueryEnd:      int(output.query_end),
		TargetStart:   int(output.target_start),
		TargetEnd:     int(output.target_end),
		Operations:    operations,
		CIGAR:         cigar,
		Matches:       int(output.matches),
		Mismatches:    int(output.mismatches),
		Insertions:    int(output.insertions),
		Deletions:     int(output.deletions),
		AlignedLength: int(output.aligned_length),
		AlignedQuery:  alignedQuery,
		AlignedTarget: alignedTarget,
	}, nil
}

func takeStringList(output *C.stride_string_list) ([]string, error) {
	defer C.stride_go_free(unsafe.Pointer(output.data))
	defer C.stride_go_free(unsafe.Pointer(output.offsets))
	count, err := checkedLength(output.count)
	if err != nil {
		return nil, err
	}
	dataLength, err := checkedLength(output.data_size)
	if err != nil {
		return nil, err
	}
	if output.offsets == nil && count != 0 {
		return nil, fmt.Errorf("stridealign: native string-list offsets are null")
	}
	var data []byte
	if dataLength != 0 {
		if output.data == nil {
			return nil, fmt.Errorf("stridealign: native string-list data is null")
		}
		data = unsafe.Slice((*byte)(unsafe.Pointer(output.data)), dataLength)
	}
	offsets := unsafe.Slice((*C.size_t)(unsafe.Pointer(output.offsets)), count+1)
	values := make([]string, count)
	for index := range values {
		begin := int(offsets[index])
		end := int(offsets[index+1])
		if begin < 0 || begin > end || end > len(data) {
			return nil, fmt.Errorf("stridealign: native string-list offsets are invalid")
		}
		values[index] = string(data[begin:end])
	}
	return values, nil
}

func nativeMatrixAvailable() ([]string, error) {
	var output C.stride_string_list
	var failure C.stride_bytes
	status := C.stride_go_matrix_available(&output, &failure)
	if err := nativeError("matrix_available", status, &failure); err != nil {
		return nil, err
	}
	return takeStringList(&output)
}

func nativeMatrix(name string) (*SubstitutionMatrix, error) {
	var output C.stride_matrix_data
	var failure C.stride_bytes
	status := C.stride_go_matrix_get(nativeString(name), &output, &failure)
	runtime.KeepAlive(name)
	if err := nativeError("matrix_get", status, &failure); err != nil {
		return nil, err
	}
	defer C.stride_go_free_matrix(&output)
	nameValue, err := copyBytes(output.name.data, output.name.size)
	if err != nil {
		return nil, err
	}
	alphabet, err := copyBytes(output.alphabet.data, output.alphabet.size)
	if err != nil {
		return nil, err
	}
	valuesLength, err := checkedLength(output.values_size)
	if err != nil {
		return nil, err
	}
	values := make([]int8, valuesLength)
	if valuesLength != 0 {
		if output.values == nil {
			return nil, fmt.Errorf("stridealign: native matrix values are null")
		}
		copy(values, unsafe.Slice((*int8)(unsafe.Pointer(output.values)), valuesLength))
	}
	return &SubstitutionMatrix{
		Name:           nameValue,
		Alphabet:       alphabet,
		WildcardIndex:  int(output.wildcard_index),
		Values:         values,
		GapScore:       int64(output.gap_score),
		GapOpenScore:   int64(output.gap_open_score),
		GapExtendScore: int64(output.gap_extend_score),
		HasAffineGaps:  output.has_affine != 0,
	}, nil
}

type nativeMatrixInput struct {
	view  C.stride_matrix_view
	name  string
	alpha string
	data  []int8
}

func makeNativeMatrix(matrix *SubstitutionMatrix) (nativeMatrixInput, error) {
	if matrix == nil {
		return nativeMatrixInput{}, fmt.Errorf("stridealign: matrix must not be nil")
	}
	input := nativeMatrixInput{
		name:  matrix.Name,
		alpha: matrix.Alphabet,
		data:  matrix.Values,
	}
	input.view.name = nativeString(input.name)
	input.view.alphabet = nativeString(input.alpha)
	input.view.wildcard_index = C.size_t(matrix.WildcardIndex)
	input.view.gap_score = C.int64_t(matrix.GapScore)
	input.view.gap_open_score = C.int64_t(matrix.GapOpenScore)
	input.view.gap_extend_score = C.int64_t(matrix.GapExtendScore)
	input.view.has_affine = C.int(boolInt(matrix.HasAffineGaps))
	if len(input.data) != 0 {
		input.view.values = (*C.int8_t)(unsafe.Pointer(&input.data[0]))
	}
	input.view.values_size = C.size_t(len(input.data))
	return input, nil
}

func (input *nativeMatrixInput) keepAlive() {
	runtime.KeepAlive(input.name)
	runtime.KeepAlive(input.alpha)
	runtime.KeepAlive(input.data)
}

func nativeMatrixStepLimit(
	matrix *SubstitutionMatrix,
	gapOpen int64,
	gapExtend int64,
) (int64, error) {
	input, err := makeNativeMatrix(matrix)
	if err != nil {
		return 0, err
	}
	var output C.int64_t
	var failure C.stride_bytes
	status := C.stride_go_matrix_score_step_limit(
		input.view, C.int64_t(gapOpen), C.int64_t(gapExtend),
		&output, &failure,
	)
	input.keepAlive()
	if err := nativeError("matrix_score_step_limit", status, &failure); err != nil {
		return 0, err
	}
	return int64(output), nil
}

func nativeMatrixEncode(matrix *SubstitutionMatrix, value string) ([]uint16, error) {
	input, err := makeNativeMatrix(matrix)
	if err != nil {
		return nil, err
	}
	var output C.stride_uints
	var failure C.stride_bytes
	status := C.stride_go_matrix_encode(
		input.view, nativeString(value), &output, &failure,
	)
	input.keepAlive()
	runtime.KeepAlive(value)
	if err := nativeError("matrix_encode", status, &failure); err != nil {
		return nil, err
	}
	defer C.stride_go_free(unsafe.Pointer(output.data))
	length, err := checkedLength(output.size)
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return []uint16{}, nil
	}
	if output.data == nil {
		return nil, fmt.Errorf("stridealign: native matrix encoding is null")
	}
	values := unsafe.Slice((*uint64)(unsafe.Pointer(output.data)), length)
	result := make([]uint16, length)
	for index, item := range values {
		if item > uint64(^uint16(0)) {
			return nil, fmt.Errorf("stridealign: native matrix token is out of range")
		}
		result[index] = uint16(item)
	}
	return result, nil
}

func nativeMatrixScore(
	local bool,
	matrix *SubstitutionMatrix,
	query string,
	target string,
	gapOpen int64,
	gapExtend int64,
) (int64, error) {
	input, err := makeNativeMatrix(matrix)
	if err != nil {
		return 0, err
	}
	var output C.int64_t
	var failure C.stride_bytes
	status := C.stride_go_matrix_score(
		C.int(boolInt(local)), input.view, nativeString(query), nativeString(target),
		C.int64_t(gapOpen), C.int64_t(gapExtend), &output, &failure,
	)
	input.keepAlive()
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	if err := nativeError("matrix_score", status, &failure); err != nil {
		return 0, err
	}
	return int64(output), nil
}

func nativeMatrixCDist(
	local bool,
	matrix *SubstitutionMatrix,
	queries []string,
	targets []string,
	gapOpen int64,
	gapExtend int64,
) ([][]float64, error) {
	input, err := makeNativeMatrix(matrix)
	if err != nil {
		return nil, err
	}
	packedQueries := packStrings(queries)
	packedTargets := packStrings(targets)
	var output C.stride_doubles
	var failure C.stride_bytes
	status := C.stride_go_matrix_cdist(
		C.int(boolInt(local)), input.view, packedQueries.view, packedTargets.view,
		C.int64_t(gapOpen), C.int64_t(gapExtend), &output, &failure,
	)
	input.keepAlive()
	packedQueries.keepAlive()
	packedTargets.keepAlive()
	if err := nativeError("matrix_cdist", status, &failure); err != nil {
		return nil, err
	}
	flattened, err := takeDoubles(&output)
	if err != nil {
		return nil, err
	}
	if len(queries) != 0 && len(targets) > int(^uint(0)>>1)/len(queries) {
		return nil, fmt.Errorf("stridealign: matrix cdist dimensions overflow")
	}
	if len(flattened) != len(queries)*len(targets) {
		return nil, fmt.Errorf("stridealign: native matrix cdist shape is inconsistent")
	}
	rows := make([][]float64, len(queries))
	for index := range rows {
		begin := index * len(targets)
		rows[index] = flattened[begin : begin+len(targets)]
	}
	return rows, nil
}

func nativeMatrixPath(
	local bool,
	matrix *SubstitutionMatrix,
	query string,
	target string,
	gapOpen int64,
	gapExtend int64,
) (AlignmentPath, error) {
	input, err := makeNativeMatrix(matrix)
	if err != nil {
		return AlignmentPath{}, err
	}
	var output C.stride_alignment_path
	var failure C.stride_bytes
	status := C.stride_go_matrix_path(
		C.int(boolInt(local)), input.view, nativeString(query), nativeString(target),
		C.int64_t(gapOpen), C.int64_t(gapExtend), &output, &failure,
	)
	input.keepAlive()
	runtime.KeepAlive(query)
	runtime.KeepAlive(target)
	if err := nativeError("matrix_path", status, &failure); err != nil {
		return AlignmentPath{}, err
	}
	return takePath(&output)
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}
