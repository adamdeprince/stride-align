#ifndef STRIDE_ALIGN_GO_BRIDGE_H
#define STRIDE_ALIGN_GO_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *data;
  size_t size;
} stride_string;

typedef struct {
  const char *data;
  size_t data_size;
  const size_t *offsets;
  size_t count;
} stride_strings;

typedef struct {
  char *data;
  size_t size;
} stride_bytes;

typedef struct {
  double *data;
  size_t size;
} stride_doubles;

typedef struct {
  const double *data;
  size_t data_size;
  const size_t *offsets;
  size_t count;
} stride_double_sequences;

typedef struct {
  uint64_t *data;
  size_t size;
} stride_uints;

typedef struct {
  double score;
  size_t index;
} stride_ranked_match;

typedef struct {
  stride_ranked_match *data;
  size_t size;
} stride_ranked_matches;

typedef struct {
  double score;
  size_t query_index;
  size_t target_index;
} stride_matrix_match;

typedef struct {
  stride_matrix_match *data;
  size_t size;
} stride_matrix_matches;

typedef struct {
  stride_ranked_match *data;
  size_t data_size;
  size_t *offsets;
  size_t query_count;
} stride_grouped_matches;

typedef struct {
  char *data;
  size_t data_size;
  size_t *offsets;
  size_t count;
} stride_string_list;

typedef struct {
  int64_t match_score;
  int64_t mismatch_score;
  int64_t gap_open_score;
  int64_t gap_extend_score;
  double prefix_weight;
  double prefix_threshold;
  size_t prefix_cap;
} stride_score_options;

typedef struct {
  int64_t score;
  size_t query_start;
  size_t query_end;
  size_t target_start;
  size_t target_end;
  stride_bytes operations;
  stride_bytes cigar;
  size_t matches;
  size_t mismatches;
  size_t insertions;
  size_t deletions;
  size_t aligned_length;
  stride_bytes aligned_query;
  stride_bytes aligned_target;
} stride_alignment_path;

typedef struct {
  stride_string name;
  stride_string alphabet;
  size_t wildcard_index;
  int64_t gap_score;
  int64_t gap_open_score;
  int64_t gap_extend_score;
  int has_affine;
  const int8_t *values;
  size_t values_size;
} stride_matrix_view;

typedef struct {
  stride_bytes name;
  stride_bytes alphabet;
  size_t wildcard_index;
  int64_t gap_score;
  int64_t gap_open_score;
  int64_t gap_extend_score;
  int has_affine;
  int8_t *values;
  size_t values_size;
} stride_matrix_data;

enum stride_pair_number_operation {
  STRIDE_PAIR_LCS_LENGTH = 0,
  STRIDE_PAIR_LCS_SUBSTRING_LENGTH = 1,
  STRIDE_PAIR_JACCARD = 2,
  STRIDE_PAIR_DICE = 3,
  STRIDE_PAIR_COSINE = 4,
  STRIDE_PAIR_OVERLAP = 5,
  STRIDE_PAIR_RATCLIFF_OBERSHELP = 6,
  STRIDE_PAIR_PARTIAL_RATIO = 7,
  STRIDE_PAIR_TOKEN_SORT_RATIO = 8,
  STRIDE_PAIR_TOKEN_SET_RATIO = 9,
  STRIDE_PAIR_PARTIAL_TOKEN_SORT_RATIO = 10,
  STRIDE_PAIR_PARTIAL_TOKEN_SET_RATIO = 11,
  STRIDE_PAIR_WRATIO = 12,
  STRIDE_PAIR_MONGE_ELKAN = 13
};

enum stride_pair_bool_operation {
  STRIDE_PAIR_SOUNDEX_EQUAL = 0,
  STRIDE_PAIR_METAPHONE_EQUAL = 1,
  STRIDE_PAIR_NYSIIS_EQUAL = 2,
  STRIDE_PAIR_MATCH_RATING_COMPARE = 3
};

enum stride_unary_string_operation {
  STRIDE_UNARY_SOUNDEX = 0,
  STRIDE_UNARY_METAPHONE = 1,
  STRIDE_UNARY_NYSIIS = 2,
  STRIDE_UNARY_MATCH_RATING_CODEX = 3,
  STRIDE_UNARY_CAVERPHONE = 4,
  STRIDE_UNARY_COLOGNE_PHONETIC = 5,
  STRIDE_UNARY_DAITCH_MOKOTOFF = 6,
  STRIDE_UNARY_BEIDER_MORSE = 7
};

const char *stride_go_version(void);
const char *stride_go_simd_level(void);
void stride_go_free(void *data);
void stride_go_free_path(stride_alignment_path *path);
void stride_go_free_matrix(stride_matrix_data *matrix);

int stride_go_score(
    int scorer,
    stride_string query,
    stride_string target,
    stride_score_options options,
    double *output,
    stride_bytes *error);

int stride_go_scores(
    int scorer,
    stride_string query,
    stride_strings targets,
    stride_score_options options,
    stride_doubles *output,
    stride_bytes *error);

int stride_go_top_k(
    int scorer,
    stride_string query,
    stride_strings targets,
    size_t k,
    stride_score_options options,
    int skip_invalid_hamming,
    stride_ranked_matches *output,
    stride_bytes *error);

int stride_go_cdist(
    int scorer,
    stride_strings queries,
    stride_strings targets,
    stride_score_options options,
    stride_doubles *output,
    stride_bytes *error);

int stride_go_cdist_above_threshold(
    int scorer,
    stride_strings queries,
    stride_strings targets,
    double threshold,
    stride_score_options options,
    stride_matrix_matches *output,
    stride_bytes *error);

int stride_go_cdist_top_k(
    int scorer,
    stride_strings queries,
    stride_strings targets,
    size_t k,
    int reject_duplicates,
    stride_score_options options,
    stride_matrix_matches *output,
    stride_bytes *error);

int stride_go_cdist_top_k_per_query(
    int scorer,
    stride_strings queries,
    stride_strings targets,
    size_t k,
    stride_score_options options,
    stride_grouped_matches *output,
    stride_bytes *error);

int stride_go_pair_number(
    int operation,
    stride_string query,
    stride_string target,
    size_t integer_option,
    stride_string string_option,
    int bool_option,
    double *output,
    stride_bytes *error);

int stride_go_pair_numbers(
    int operation,
    stride_string query,
    stride_strings targets,
    size_t integer_option,
    stride_string string_option,
    int bool_option,
    stride_doubles *output,
    stride_bytes *error);

int stride_go_lcs_substring(
    stride_string query,
    stride_string target,
    stride_bytes *output,
    stride_bytes *error);

int stride_go_lcs_substrings(
    stride_string query,
    stride_strings targets,
    stride_string_list *output,
    stride_bytes *error);

int stride_go_pair_bool(
    int operation,
    stride_string query,
    stride_string target,
    int variant,
    int *output,
    stride_bytes *error);

int stride_go_pair_bools(
    int operation,
    stride_string query,
    stride_strings targets,
    int variant,
    stride_uints *output,
    stride_bytes *error);

int stride_go_unary_string(
    int operation,
    stride_string input,
    int64_t integer_option_a,
    int64_t integer_option_b,
    int bool_option_a,
    int bool_option_b,
    stride_bytes *output,
    stride_bytes *error);

int stride_go_unary_strings(
    int operation,
    stride_strings inputs,
    int64_t integer_option_a,
    int64_t integer_option_b,
    int bool_option_a,
    int bool_option_b,
    stride_string_list *output,
    stride_bytes *error);

int stride_go_double_metaphone(
    stride_string input,
    size_t maximum_length,
    int variant,
    stride_bytes *primary,
    stride_bytes *alternate,
    stride_bytes *error);

int stride_go_double_metaphones(
    stride_strings inputs,
    size_t maximum_length,
    int variant,
    stride_string_list *primaries,
    stride_string_list *alternates,
    stride_bytes *error);

int stride_go_dtw(
    const double *query,
    size_t query_size,
    const double *target,
    size_t target_size,
    double window,
    int distance_kind,
    double score_cutoff,
    double *output,
    stride_bytes *error);

int stride_go_dtw_distances(
    const double *query,
    size_t query_size,
    stride_double_sequences targets,
    double window,
    int distance_kind,
    double score_cutoff,
    stride_doubles *output,
    stride_bytes *error);

int stride_go_alignment_path(
    int local,
    stride_string query,
    stride_string target,
    stride_score_options options,
    stride_alignment_path *output,
    stride_bytes *error);

int stride_go_matrix_available(
    stride_string_list *output,
    stride_bytes *error);

int stride_go_matrix_get(
    stride_string name,
    stride_matrix_data *output,
    stride_bytes *error);

int stride_go_matrix_score_step_limit(
    stride_matrix_view matrix,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    int64_t *output,
    stride_bytes *error);

int stride_go_matrix_encode(
    stride_matrix_view matrix,
    stride_string input,
    stride_uints *output,
    stride_bytes *error);

int stride_go_matrix_score(
    int local,
    stride_matrix_view matrix,
    stride_string query,
    stride_string target,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    int64_t *output,
    stride_bytes *error);

int stride_go_matrix_cdist(
    int local,
    stride_matrix_view matrix,
    stride_strings queries,
    stride_strings targets,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    stride_doubles *output,
    stride_bytes *error);

int stride_go_matrix_path(
    int local,
    stride_matrix_view matrix,
    stride_string query,
    stride_string target,
    int64_t gap_open_score,
    int64_t gap_extend_score,
    stride_alignment_path *output,
    stride_bytes *error);

#ifdef __cplusplus
}
#endif

#endif
