# DuckDB SQL function catalog

The DuckDB v1.5.5 stride-align package registers 167 SQL function names with
413 overloads. Every endpoint below accepts literals or columns of the listed
DuckDB types. RapidFuzz, Parasail, Jellyfish, and TheFuzz compatibility shims
are intentionally absent.

Use DuckDB itself for the authoritative signature and return-type list for the
package you loaded:

```sql
SELECT function_name, parameters, parameter_types, return_type
FROM duckdb_functions()
WHERE function_name LIKE 'stride_%'
ORDER BY function_name, parameter_types;
```

## Package and backend

- `stride_align_simd_level`
- `stride_available_backends`
- `stride_backend_is_available`
- `stride_detect_best_backend`

## Pairwise distances and similarities

- `stride_levenshtein`, `stride_levenshtein_score`, `stride_levenshtein_similarity`, `stride_levenshtein_normalized_score`
- `stride_osa`, `stride_osa_similarity`, `stride_damerau_levenshtein_score`, `stride_damerau_levenshtein_normalized_score`
- `stride_true_damerau_levenshtein`, `stride_true_damerau_levenshtein_score`, `stride_true_damerau_levenshtein_similarity`, `stride_true_damerau_levenshtein_normalized_score`
- `stride_indel`, `stride_indel_score`, `stride_indel_similarity`, `stride_indel_normalized_score`
- `stride_hamming`, `stride_hamming_score`, `stride_hamming_similarity`, `stride_hamming_normalized_score`
- `stride_jaro`, `stride_jaro_similarity`
- `stride_jaro_winkler`, `stride_jaro_winkler_similarity`

## Pairwise alignment scores

- `stride_smith_waterman`, `stride_smith_waterman_affine`, `stride_smith_waterman_score`, `stride_smith_waterman_normalized_score`
- `stride_smith_waterman_farrar_score`, `stride_smith_waterman_farrar_normalized_score`
- `stride_needleman_wunsch`, `stride_needleman_wunsch_affine`, `stride_needleman_wunsch_score`, `stride_needleman_wunsch_normalized_score`

## One-to-many vectors

- `stride_scores`
- `stride_levenshtein_scores`, `stride_levenshtein_normalized_scores`
- `stride_damerau_levenshtein_scores`, `stride_damerau_levenshtein_normalized_scores`
- `stride_true_damerau_levenshtein_scores`, `stride_true_damerau_levenshtein_normalized_scores`
- `stride_indel_scores`, `stride_indel_normalized_scores`
- `stride_hamming_scores`, `stride_hamming_normalized_scores`
- `stride_jaro_similarities`, `stride_jaro_winkler_similarities`
- `stride_smith_waterman_scores`, `stride_smith_waterman_normalized_scores`
- `stride_smith_waterman_farrar_scores`, `stride_smith_waterman_farrar_normalized_scores`
- `stride_needleman_wunsch_scores`, `stride_needleman_wunsch_normalized_scores`

## Best-match and top-k ranking

- `stride_extract`, `stride_extract_best`
- `stride_levenshtein_top_k`, `stride_levenshtein_best`, `stride_levenshtein_normalized_top_k`, `stride_levenshtein_normalized_best`
- `stride_damerau_levenshtein_top_k`, `stride_damerau_levenshtein_best`, `stride_damerau_levenshtein_normalized_top_k`, `stride_damerau_levenshtein_normalized_best`
- `stride_true_damerau_levenshtein_top_k`, `stride_true_damerau_levenshtein_best`, `stride_true_damerau_levenshtein_normalized_top_k`, `stride_true_damerau_levenshtein_normalized_best`
- `stride_indel_top_k`, `stride_indel_best`, `stride_indel_normalized_top_k`, `stride_indel_normalized_best`
- `stride_hamming_top_k`, `stride_hamming_best`, `stride_hamming_normalized_top_k`, `stride_hamming_normalized_best`
- `stride_jaro_top_k`, `stride_jaro_best`
- `stride_jaro_winkler_top_k`, `stride_jaro_winkler_best`
- `stride_smith_waterman_top_k`, `stride_smith_waterman_best`

## All-pairs operations

- `stride_cdist`
- `stride_cdist_above_threshold`
- `stride_cdist_top_k`
- `stride_cdist_top_k_per_query`

## Text, subsequence, and token algorithms

- `stride_lcs_length`, `stride_lcs_substring_length`, `stride_lcs_substring`
- `stride_jaccard`, `stride_jaccard_similarities`
- `stride_dice`, `stride_dice_similarities`
- `stride_cosine`, `stride_cosine_similarities`
- `stride_overlap`, `stride_overlap_similarities`
- `stride_ratcliff_obershelp_similarity`, `stride_ratcliff_obershelp_similarities`
- `stride_partial_ratio`, `stride_partial_ratios`
- `stride_token_sort_ratio`, `stride_token_sort_ratios`
- `stride_token_set_ratio`, `stride_token_set_ratios`
- `stride_partial_token_sort_ratio`, `stride_partial_token_sort_ratios`
- `stride_partial_token_set_ratio`, `stride_partial_token_set_ratios`
- `stride_wratio`, `stride_wratios`
- `stride_monge_elkan`

## Phonetic algorithms

- `stride_soundex`, `stride_soundex_equal`
- `stride_metaphone`, `stride_metaphone_equal`
- `stride_nysiis`, `stride_nysiis_equal`
- `stride_match_rating_codex`, `stride_match_rating_compare`
- `stride_caverphone`
- `stride_cologne_phonetic`
- `stride_daitch_mokotoff`
- `stride_double_metaphone`
- `stride_beider_morse`

## Dynamic Time Warping

- `stride_dtw`
- `stride_dtw_distances`

## Alignment paths and CIGARs

- `stride_smith_waterman_path`, `stride_smith_waterman_path_info`, `stride_smith_waterman_cigar`, `stride_smith_waterman_trace_cigar`, `stride_smith_waterman_trade_cigar`
- `stride_needleman_wunsch_path`, `stride_needleman_wunsch_path_info`, `stride_needleman_wunsch_cigar`, `stride_needleman_wunsch_trace_cigar`, `stride_needleman_wunsch_trade_cigar`

## Matrix construction and inspection

- `stride_substitution_matrix`, `stride_ascii_matrix`, `stride_identity_matrix`
- `stride_matrix_available`, `stride_matrix_info`, `stride_matrix_encode`, `stride_matrix_transpose`
- `stride_matrix_from_ncbi_text`, `stride_matrix_score_step_limit`
- `stride_substitution_matrix_score`
- `stride_keyboard_available`, `stride_keyboard_from_confusion_counts`, `stride_keyboard_from_npy`

## Matrix alignment and all-pairs operations

- `stride_smith_waterman_matrix_score`, `stride_smith_waterman_matrix_scores`
- `stride_smith_waterman_matrix_path`, `stride_smith_waterman_matrix_path_info`, `stride_smith_waterman_matrix_cigar`
- `stride_needleman_wunsch_matrix_score`, `stride_needleman_wunsch_matrix_scores`
- `stride_needleman_wunsch_matrix_path`, `stride_needleman_wunsch_matrix_path_info`, `stride_needleman_wunsch_matrix_cigar`
- `stride_cdist_matrix_local`, `stride_cdist_matrix_global`
- `stride_cdist_matrix_above_threshold`, `stride_cdist_matrix_top_k`
