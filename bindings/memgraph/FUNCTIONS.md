# Memgraph function catalog

The module registers 90 scalar Cypher functions plus the streaming `cdist`
procedure. Calls use the module namespace, for example
`stride_align.levenshtein(query, target)`.

## Package

- `version`
- `simd_level`, `detect_best_backend`, `backend_is_available`

## Pairwise distances and similarities

- `levenshtein`, `levenshtein_score`, `levenshtein_similarity`, `levenshtein_normalized_score`
- `osa`, `osa_similarity`, `damerau_levenshtein_score`, `damerau_levenshtein_normalized_score`
- `true_damerau_levenshtein`, `true_damerau_levenshtein_score`, `true_damerau_levenshtein_similarity`, `true_damerau_levenshtein_normalized_score`
- `indel`, `indel_score`, `indel_similarity`, `indel_normalized_score`
- `hamming`, `hamming_score`, `hamming_similarity`, `hamming_normalized_score`
- `jaro`, `jaro_similarity`
- `jaro_winkler`, `jaro_winkler_similarity`
- `score` for a scorer selected by canonical string name

## Pairwise alignment scores

- `smith_waterman`, `smith_waterman_affine`
- `smith_waterman_score`, `smith_waterman_normalized_score`
- `smith_waterman_farrar_score`, `smith_waterman_farrar_normalized_score`
- `needleman_wunsch`, `needleman_wunsch_affine`
- `needleman_wunsch_score`, `needleman_wunsch_normalized_score`

## Text, subsequence, and token algorithms

- `lcs_length`, `lcs_substring_length`, `lcs_substring`
- `jaccard`, `dice`, `cosine`, `overlap`
- `ratcliff_obershelp_similarity`
- `partial_ratio`, `token_sort_ratio`, `token_set_ratio`
- `partial_token_sort_ratio`, `partial_token_set_ratio`
- `wratio` and the Python-spelled alias `WRatio`
- `monge_elkan`

## Phonetic algorithms

- `soundex`, `soundex_equal`
- `metaphone`, `metaphone_equal`
- `nysiis`, `nysiis_equal`
- `match_rating_codex`, `match_rating_compare`
- `caverphone`, `cologne_phonetic`
- `daitch_mokotoff`, `double_metaphone`, `beider_morse`

## Dynamic time warping

- `dtw` returns one distance for two numeric sequences

## Alignment paths and CIGARs

- `smith_waterman_path`, `smith_waterman_path_info`
- `smith_waterman_cigar`, `smith_waterman_trace_cigar`, `smith_waterman_trade_cigar`
- `needleman_wunsch_path`, `needleman_wunsch_path_info`
- `needleman_wunsch_cigar`, `needleman_wunsch_trace_cigar`, `needleman_wunsch_trade_cigar`

## Built-in substitution matrices

- `matrix_info`, `matrix_score_step_limit`, `substitution_matrix_score`
- `smith_waterman_matrix_score`, `smith_waterman_matrix_path`, `smith_waterman_matrix_path_info`, `smith_waterman_matrix_cigar`
- `needleman_wunsch_matrix_score`, `needleman_wunsch_matrix_path`, `needleman_wunsch_matrix_path_info`, `needleman_wunsch_matrix_cigar`

Built-in names include the BLOSUM, PAM, NUC.4.4, DNA match, ASCII, and QWERTY
catalogs shipped by stride-align. List-producing matrix catalogs, custom matrix
constructors, matrix batches, and matrix `cdist` variants are not part of the
Memgraph scalar surface.

## Streaming procedure

- `cdist(queries, targets, scorer, ...)`

It yields `query_index`, `target_index`, and `score`, one logical row per pair.

## Deliberately absent

- plural one-to-many functions such as `levenshtein_scores`
- `best`, `extract`, and `top_k` families
- threshold and top-k `cdist` variants
- `dtw_distances`
- list-returning matrix catalogs and matrix batches
- RapidFuzz, TheFuzz, Parasail, and Jellyfish compatibility shims
