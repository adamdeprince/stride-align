# ruff: noqa: E501
"""Cross-language examples used by the generated API landing page."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class ApiExample:
    slug: str
    family: str
    title: str
    description: str
    python: str
    r: str
    duckdb: str
    detail: str | None = None


@dataclass(frozen=True)
class VectorizationExample:
    slug: str
    title: str
    input_shape: str
    output_shape: str
    description: str
    python: str
    r: str
    duckdb: str


VECTORIZATION_EXAMPLES = (
    VectorizationExample(
        "shape-one-to-many",
        "One query against many targets",
        "1 query + T targets",
        "T scores",
        "A score is returned for every target, in target order. This is the direct batch shape when one query is compared with a candidate collection.",
        'targets = ["sitting", "bitten", "kitchen"]\nscores = sa.levenshtein_normalized_scores("kitten", targets)\nassert scores.shape == (3,)\n# scores[i] belongs to targets[i].',
        'targets <- c("sitting", "bitten", "kitchen")\nscores <- levenshtein_normalized_scores("kitten", targets)\nstopifnot(length(scores) == 3L)\n# scores[[i]] belongs to targets[[i]].',
        "SELECT stride_levenshtein_normalized_scores(\n  'kitten', ['sitting', 'bitten', 'kitchen']\n) AS scores;",
    ),
    VectorizationExample(
        "shape-ranking",
        "Best match and one-to-many top-k",
        "1 query + T targets + K",
        "1 match or at most K matches",
        "best returns one candidate; top_k retains only the requested candidates, so application code does not have to construct and sort a complete score vector.",
        'targets = ["sitting", "bitten", "kitchen"]\nbest = sa.levenshtein_normalized_best("kitten", targets)\ntop = sa.levenshtein_normalized_top_k("kitten", targets, k=2)\nassert best[2] == targets.index(best[0])\nassert len(top) == 2\n# Each Python match is (target, score, zero_based_index).',
        'targets <- c("sitting", "bitten", "kitchen")\nbest <- levenshtein_normalized_best("kitten", targets)\ntop <- levenshtein_normalized_top_k("kitten", targets, k = 2)\nstopifnot(best$target == targets[[best$index]])\nstopifnot(nrow(top) == 2L)\n# best is a named list; top is a data frame with one-based index.',
        "SELECT\n  stride_levenshtein_normalized_best(\n    'kitten', ['sitting', 'bitten', 'kitchen']\n  ) AS best,\n  stride_levenshtein_normalized_top_k(\n    'kitten', ['sitting', 'bitten', 'kitchen'], 2\n  ) AS top;",
    ),
    VectorizationExample(
        "shape-cdist-matrix",
        "Dense all-pairs matrix",
        "Q queries × T targets",
        "Q rows × T columns",
        "cdist scores the Cartesian product. Row q belongs to queries[q], column t belongs to targets[t], and cell [q, t] is that pair's score.",
        'queries = ["kitten", "sitting"]\ntargets = ["kitten", "bitten", "kitchen"]\nscores = sa.cdist(queries, targets, scorer=sa.Scorer.JARO)\nassert scores.shape == (2, 3)\nquery_0_target_1 = scores[0, 1]',
        'queries <- c("kitten", "sitting")\ntargets <- c("kitten", "bitten", "kitchen")\nscores <- cdist(queries, targets, scorer = Scorer$JARO)\nstopifnot(identical(dim(scores), c(2L, 3L)))\nquery_1_target_2 <- scores[1, 2]',
        "WITH scored AS (\n  SELECT stride_cdist(\n    ['kitten', 'sitting'],\n    ['kitten', 'bitten', 'kitchen'],\n    'jaro'\n  ) AS scores\n)\nSELECT\n  scores,\n  len(scores) AS query_count,\n  len(scores[1]) AS target_count,\n  scores[1][2] AS query_0_target_1\nFROM scored;",
    ),
    VectorizationExample(
        "shape-global-selection",
        "Threshold filtering and global top-k",
        "Q × T conceptual grid",
        "qualifying pair records or at most K total",
        "The threshold form emits every qualifying pair. Global cdist_top_k holds one competition across the entire matrix, so k=2 means two matches total; some queries may have no returned match.",
        'queries = ["kitten", "sitting"]\ntargets = ["kitten", "bitten", "kitchen"]\nfiltered = list(sa.cdist_above_threshold(\n    queries, targets, scorer=sa.Scorer.JARO, threshold=0.8\n))\nglobal_top = sa.cdist_top_k(\n    queries, targets, scorer=sa.Scorer.JARO, k=2\n)\nassert len(global_top) <= 2\n# Python records are (score, query, target).',
        'queries <- c("kitten", "sitting")\ntargets <- c("kitten", "bitten", "kitchen")\nfiltered <- cdist_above_threshold(\n  queries, targets, scorer = Scorer$JARO, threshold = 0.8\n)\nglobal_top <- cdist_top_k(\n  queries, targets, scorer = Scorer$JARO, k = 2\n)\nstopifnot(nrow(global_top) <= 2L)\n# Both results are data frames with one-based source indices.',
        "SELECT stride_cdist_above_threshold(\n  ['kitten', 'sitting'],\n  ['kitten', 'bitten', 'kitchen'],\n  'jaro', 0.8\n) AS filtered;\nSELECT stride_cdist_top_k(\n  ['kitten', 'sitting'],\n  ['kitten', 'bitten', 'kitchen'],\n  'jaro', 2\n) AS global_top;",
    ),
    VectorizationExample(
        "shape-per-query-selection",
        "Top-k for each query",
        "Q queries × T targets + K",
        "Q groups, each with at most K matches",
        "cdist_top_k_per_query runs a separate competition for every query. With k=2 it can return up to 2Q matches, and every nonempty query row gets its own nearest-neighbor list.",
        'queries = ["kitten", "sitting"]\ntargets = ["kitten", "bitten", "kitchen"]\nper_query = list(sa.cdist_top_k_per_query(\n    queries, targets, scorer=sa.Scorer.JARO, k=2\n))\nassert len(per_query) == len(queries)\nassert all(len(matches) <= 2 for _, matches in per_query)\n# Each item is (query, [(score, target), ...]).',
        'queries <- c("kitten", "sitting")\ntargets <- c("kitten", "bitten", "kitchen")\nper_query <- cdist_top_k_per_query(\n  queries, targets, scorer = Scorer$JARO, k = 2\n)\nstopifnot(length(per_query) == length(queries))\nstopifnot(all(vapply(per_query, nrow, integer(1)) <= 2L))\n# A named list of data frames; source indices are one-based.',
        "SELECT stride_cdist_top_k_per_query(\n  ['kitten', 'sitting'],\n  ['kitten', 'bitten', 'kitchen'],\n  'jaro', 2\n) AS per_query;",
    ),
)


EXAMPLES = (
    ApiExample(
        "levenshtein",
        "Edit distance",
        "Levenshtein",
        "Insertions, deletions, and substitutions.",
        'sa.levenshtein_normalized_score("kitten", "sitting")',
        'levenshtein_normalized_score("kitten", "sitting")',
        "SELECT stride_levenshtein_normalized_score('kitten', 'sitting');",
        "edit-distance.html",
    ),
    ApiExample(
        "osa",
        "Edit distance",
        "Damerau–Levenshtein (OSA)",
        "Levenshtein distance plus one-use adjacent transpositions.",
        'sa.damerau_levenshtein_score("CA", "AC")',
        'damerau_levenshtein_score("CA", "AC")',
        "SELECT stride_damerau_levenshtein_score('CA', 'AC');",
        "edit-distance.html",
    ),
    ApiExample(
        "true-damerau",
        "Edit distance",
        "Unrestricted Damerau–Levenshtein",
        "Damerau's unrestricted edit distance for overlapping transpositions.",
        'sa.true_damerau_levenshtein_score("CA", "ABC")',
        'true_damerau_levenshtein_score("CA", "ABC")',
        "SELECT stride_true_damerau_levenshtein_score('CA', 'ABC');",
        "edit-distance.html",
    ),
    ApiExample(
        "indel",
        "Edit distance",
        "Indel",
        "Insert-and-delete distance, with substitutions represented as two edits.",
        'sa.indel_normalized_score("fuzzy", "fizzi")',
        'indel_normalized_score("fuzzy", "fizzi")',
        "SELECT stride_indel_normalized_score('fuzzy', 'fizzi');",
        "edit-distance.html",
    ),
    ApiExample(
        "hamming",
        "Edit distance",
        "Hamming",
        "Position-wise substitutions for equal-length strings.",
        'sa.hamming_score("karolin", "kathrin")',
        'hamming_score("karolin", "kathrin")',
        "SELECT stride_hamming_score('karolin', 'kathrin');",
        "edit-distance.html",
    ),
    ApiExample(
        "jaro",
        "Similarity",
        "Jaro",
        "Matching-window similarity with a transposition penalty.",
        'sa.jaro_similarity("MARTHA", "MARHTA")',
        'jaro_similarity("MARTHA", "MARHTA")',
        "SELECT stride_jaro_similarity('MARTHA', 'MARHTA');",
        "similarity.html",
    ),
    ApiExample(
        "jaro-winkler",
        "Similarity",
        "Jaro–Winkler",
        "Jaro similarity with a configurable common-prefix bonus.",
        'sa.jaro_winkler_similarity("MARTHA", "MARHTA", prefix_weight=0.1)',
        'jaro_winkler_similarity("MARTHA", "MARHTA", prefix_weight = 0.1)',
        "SELECT stride_jaro_winkler_similarity('MARTHA', 'MARHTA', 0.1, 0.7, 4);",
        "similarity.html",
    ),
    ApiExample(
        "lcs",
        "Sequence comparison",
        "Longest common subsequence and substring",
        "Lengths and the actual longest contiguous common substring.",
        'sa.lcs_length("banana", "ananas")\nsa.lcs_substring_length("banana", "ananas")\nsa.lcs_substring("banana", "ananas")',
        'lcs_length("banana", "ananas")\nlcs_substring_length("banana", "ananas")\nlcs_substring("banana", "ananas")',
        "SELECT stride_lcs_length('banana', 'ananas');\nSELECT stride_lcs_substring_length('banana', 'ananas');\nSELECT stride_lcs_substring('banana', 'ananas');",
        "similarity.html",
    ),
    ApiExample(
        "ngrams",
        "Similarity",
        "Character n-grams",
        "Jaccard, Dice, cosine, and overlap similarity over character n-grams.",
        'sa.jaccard("night", "nacht", n=2)\nsa.dice("night", "nacht", n=2)\nsa.cosine("night", "nacht", n=2)\nsa.overlap("night", "nacht", n=2)',
        'jaccard("night", "nacht", n = 2)\ndice("night", "nacht", n = 2)\ncosine("night", "nacht", n = 2)\noverlap("night", "nacht", n = 2)',
        "SELECT stride_jaccard('night', 'nacht', 2);\nSELECT stride_dice('night', 'nacht', 2);\nSELECT stride_cosine('night', 'nacht', 2);\nSELECT stride_overlap('night', 'nacht', 2);",
        "similarity.html",
    ),
    ApiExample(
        "ratcliff-obershelp",
        "Similarity",
        "Ratcliff–Obershelp",
        "Recursive longest-common-substring similarity.",
        'sa.ratcliff_obershelp_similarity("abcd", "abdc")',
        'ratcliff_obershelp_similarity("abcd", "abdc")',
        "SELECT stride_ratcliff_obershelp_similarity('abcd', 'abdc');",
        "similarity.html",
    ),
    ApiExample(
        "token-ratios",
        "Similarity",
        "Partial, token, and weighted ratios",
        "Substring and token-order tolerant native scorers, including WRatio.",
        'sa.partial_ratio("New York Mets", "Mets")\nsa.token_sort_ratio("New York Mets", "Mets New York")\nsa.token_set_ratio("New York Mets", "New York Mets vs Braves")\nsa.partial_token_sort_ratio("New York Mets", "Mets New York")\nsa.partial_token_set_ratio("New York Mets", "Mets vs Braves")\nsa.WRatio("New York Mets", "New York Meats")',
        'partial_ratio("New York Mets", "Mets")\ntoken_sort_ratio("New York Mets", "Mets New York")\ntoken_set_ratio("New York Mets", "New York Mets vs Braves")\npartial_token_sort_ratio("New York Mets", "Mets New York")\npartial_token_set_ratio("New York Mets", "Mets vs Braves")\nWRatio("New York Mets", "New York Meats")',
        "SELECT stride_partial_ratio('New York Mets', 'Mets');\nSELECT stride_token_sort_ratio('New York Mets', 'Mets New York');\nSELECT stride_token_set_ratio('New York Mets', 'New York Mets vs Braves');\nSELECT stride_partial_token_sort_ratio('New York Mets', 'Mets New York');\nSELECT stride_partial_token_set_ratio('New York Mets', 'Mets vs Braves');\nSELECT stride_wratio('New York Mets', 'New York Meats');",
        "similarity.html",
    ),
    ApiExample(
        "monge-elkan",
        "Similarity",
        "Monge–Elkan",
        "Best-token composite similarity for multi-word fields.",
        'sa.monge_elkan("Computer Science Department", "Department of Computer Science", symmetric=True)',
        'monge_elkan("Computer Science Department", "Department of Computer Science", symmetric = TRUE)',
        "SELECT stride_monge_elkan('Computer Science Department', 'Department of Computer Science', 'jaro', true);",
        "similarity.html",
    ),
    ApiExample(
        "smith-waterman",
        "Alignment",
        "Smith–Waterman",
        "Local sequence alignment with linear or affine gaps.",
        'sa.smith_waterman_score("ACACACTA", "AGCACACA", gap_score=-1)',
        'smith_waterman_score("ACACACTA", "AGCACACA", gap_score = -1)',
        "SELECT stride_smith_waterman_score('ACACACTA', 'AGCACACA', 2, -1, -1);",
        "alignment.html",
    ),
    ApiExample(
        "needleman-wunsch",
        "Alignment",
        "Needleman–Wunsch",
        "Global sequence alignment with linear or affine gaps.",
        'sa.needleman_wunsch_score("GATTACA", "GCATGCU", gap_score=-1)',
        'needleman_wunsch_score("GATTACA", "GCATGCU", gap_score = -1)',
        "SELECT stride_needleman_wunsch_score('GATTACA', 'GCATGCU', 2, -1, -1);",
        "alignment.html",
    ),
    ApiExample(
        "traceback",
        "Alignment",
        "Alignment paths and CIGAR",
        "Traceback metadata, aligned strings, and standard or extended CIGAR output.",
        'sa.smith_waterman_cigar("ACGTAC", "ACGTAG")',
        'smith_waterman_cigar("ACGTAC", "ACGTAG")',
        "SELECT stride_smith_waterman_cigar('ACGTAC', 'ACGTAG');",
        "alignment.html",
    ),
    ApiExample(
        "one-to-many",
        "Batch operations",
        "One-to-many and best match",
        "Score a candidate vector or run the bounded top-one operation.",
        'targets = ["sitting", "bitten", "kitchen"]\nsa.levenshtein_normalized_scores("kitten", targets)\nsa.levenshtein_normalized_best("kitten", targets)\nsa.levenshtein_normalized_top_k("kitten", targets, k=2)',
        'targets <- c("sitting", "bitten", "kitchen")\nlevenshtein_normalized_scores("kitten", targets)\nlevenshtein_normalized_best("kitten", targets)\nlevenshtein_normalized_top_k("kitten", targets, k = 2)',
        "SELECT stride_levenshtein_normalized_scores('kitten', ['sitting', 'bitten', 'kitchen']);\nSELECT stride_levenshtein_normalized_best('kitten', ['sitting', 'bitten', 'kitchen']);\nSELECT stride_levenshtein_normalized_top_k('kitten', ['sitting', 'bitten', 'kitchen'], 2);",
        "edit-distance.html",
    ),
    ApiExample(
        "cdist",
        "Batch operations",
        "All-pairs cdist and top-k",
        "Full matrices, threshold filters, global top-k, and top-k per query.",
        'queries = ["Martha", "Arthur"]\ntargets = ["Martha", "Marhta"]\nsa.cdist(queries, targets, scorer=sa.Scorer.JARO_WINKLER)\nsa.cdist_top_k_per_query(queries, targets, scorer=sa.Scorer.JARO_WINKLER, k=1)',
        'queries <- c("Martha", "Arthur")\ntargets <- c("Martha", "Marhta")\ncdist(queries, targets, scorer = Scorer$JARO_WINKLER)\ncdist_top_k_per_query(queries, targets, scorer = Scorer$JARO_WINKLER, k = 1)',
        "SELECT stride_cdist(['Martha', 'Arthur'], ['Martha', 'Marhta'], 'jaro_winkler');\nSELECT stride_cdist_top_k_per_query(['Martha', 'Arthur'], ['Martha', 'Marhta'], 'jaro_winkler', 1);",
        "cdist.html",
    ),
    ApiExample(
        "matrices",
        "Alignment",
        "Substitution matrices",
        "Built-in BLOSUM, PAM, nucleotide, text, and custom scoring matrices.",
        'from stride_align.matrices import blosum62\n\nsa.smith_waterman_score("HEAGAWGHEE", "PAWHEAE", matrix=blosum62)',
        'smith_waterman_score("HEAGAWGHEE", "PAWHEAE", matrix = blosum62)',
        "SELECT stride_smith_waterman_matrix_score('HEAGAWGHEE', 'PAWHEAE', 'blosum62');",
        "matrices.html",
    ),
    ApiExample(
        "dtw",
        "Time series",
        "Dynamic Time Warping",
        "Elastic alignment for numeric sequences with an optional Sakoe–Chiba window.",
        "import numpy as np\n\nquery = np.array([1.0, 2.0, 3.0])\ntarget = np.array([1.0, 2.5, 3.0])\nsa.dtw(query, target, window=1)",
        "dtw(c(1, 2, 3), c(1, 2.5, 3), window = 1)",
        "SELECT stride_dtw([1.0, 2.0, 3.0], [1.0, 2.5, 3.0], 1);",
        "dtw.html",
    ),
    ApiExample(
        "soundex",
        "Phonetics",
        "Soundex",
        "Compact English-language phonetic keys.",
        'sa.soundex("Robert")',
        'soundex("Robert")',
        "SELECT stride_soundex('Robert');",
        "phonetic.html",
    ),
    ApiExample(
        "metaphone",
        "Phonetics",
        "Metaphone",
        "English phonetic encoding with Philips and Jellyfish rule variants.",
        'sa.metaphone("Schwarzenegger")',
        'metaphone("Schwarzenegger")',
        "SELECT stride_metaphone('Schwarzenegger');",
        "phonetic.html",
    ),
    ApiExample(
        "double-metaphone",
        "Phonetics",
        "Double Metaphone",
        "Primary and alternate pronunciation keys.",
        'sa.double_metaphone("Smith")',
        'double_metaphone("Smith")',
        "SELECT stride_double_metaphone('Smith');",
        "phonetic.html",
    ),
    ApiExample(
        "nysiis",
        "Phonetics",
        "NYSIIS",
        "New York State Identification and Intelligence System keys.",
        'sa.nysiis("Macdonald")',
        'nysiis("Macdonald")',
        "SELECT stride_nysiis('Macdonald');",
        "phonetic.html",
    ),
    ApiExample(
        "match-rating",
        "Phonetics",
        "Match Rating Approach",
        "Codex generation and direct name comparison.",
        'sa.match_rating_compare("Smith", "Smyth")',
        'match_rating_compare("Smith", "Smyth")',
        "SELECT stride_match_rating_compare('Smith', 'Smyth');",
        "phonetic.html",
    ),
    ApiExample(
        "caverphone",
        "Phonetics",
        "Caverphone 2.0",
        "English name keys designed for New Zealand records.",
        'sa.caverphone("Catherine")',
        'caverphone("Catherine")',
        "SELECT stride_caverphone('Catherine');",
        "phonetic.html",
    ),
    ApiExample(
        "cologne",
        "Phonetics",
        "Cologne Phonetic",
        "Unicode-aware German phonetic encoding.",
        'sa.cologne_phonetic("Müller")',
        'cologne_phonetic("Müller")',
        "SELECT stride_cologne_phonetic('Müller');",
        "phonetic.html",
    ),
    ApiExample(
        "daitch-mokotoff",
        "Phonetics",
        "Daitch–Mokotoff Soundex",
        "Branching six-digit keys for Slavic and Yiddish surnames.",
        'sa.daitch_mokotoff("Goldberg")',
        'daitch_mokotoff("Goldberg")',
        "SELECT stride_daitch_mokotoff('Goldberg');",
        "phonetic.html",
    ),
    ApiExample(
        "beider-morse",
        "Phonetics",
        "Beider–Morse",
        "Multi-language phonetic matching for genealogical names.",
        'sa.beider_morse("Müller")',
        'beider_morse("Müller")',
        "SELECT stride_beider_morse('Müller');",
        "phonetic.html",
    ),
)


ZH_EXAMPLE_TEXT = {
    "levenshtein": ("编辑距离", "Levenshtein 距离", "插入、删除与替换。"),
    "osa": (
        "编辑距离",
        "Damerau–Levenshtein（OSA）",
        "Levenshtein 距离加上每个字符至多参与一次的相邻换位。",
    ),
    "true-damerau": (
        "编辑距离",
        "无限制 Damerau–Levenshtein",
        "允许换位重叠的无限制 Damerau 编辑距离。",
    ),
    "indel": ("编辑距离", "插入删除距离", "仅使用插入和删除；一次替换计作两次编辑。"),
    "hamming": ("编辑距离", "Hamming 距离", "用于等长字符串的逐位置替换距离。"),
    "jaro": ("相似度", "Jaro 相似度", "采用匹配窗口并对换位施加惩罚的相似度。"),
    "jaro-winkler": ("相似度", "Jaro–Winkler 相似度", "在 Jaro 相似度上增加可配置的共同前缀奖励。"),
    "lcs": ("序列比较", "最长公共子序列与子串", "返回长度以及实际的最长连续公共子串。"),
    "ngrams": ("相似度", "字符 n 元组", "基于字符 n 元组的 Jaccard、Dice、余弦与重叠相似度。"),
    "ratcliff-obershelp": ("相似度", "Ratcliff–Obershelp 相似度", "递归计算最长公共子串的相似度。"),
    "token-ratios": (
        "相似度",
        "部分、词元与加权比率",
        "可容忍子串位置和词元顺序变化的原生评分器，包括 WRatio。",
    ),
    "monge-elkan": ("相似度", "Monge–Elkan 相似度", "用于多词字段的最佳词元组合相似度。"),
    "smith-waterman": ("序列比对", "Smith–Waterman 局部比对", "使用线性或仿射空位的局部序列比对。"),
    "needleman-wunsch": (
        "序列比对",
        "Needleman–Wunsch 全局比对",
        "使用线性或仿射空位的全局序列比对。",
    ),
    "traceback": (
        "序列比对",
        "比对路径与 CIGAR",
        "返回回溯元数据、对齐后的字符串以及标准或扩展 CIGAR。",
    ),
    "one-to-many": ("批量操作", "一对多与最佳匹配", "为候选向量评分，或执行有界的最佳一项筛选。"),
    "cdist": (
        "批量操作",
        "全配对距离矩阵与前 k 项",
        "稠密矩阵、阈值过滤、全局前 k 项以及每个查询各自的前 k 项。",
    ),
    "matrices": ("序列比对", "替换矩阵", "内置 BLOSUM、PAM、核苷酸、文本及自定义评分矩阵。"),
    "dtw": ("时间序列", "动态时间规整", "用于数值序列的弹性比对，可选 Sakoe–Chiba 窗口。"),
    "soundex": ("语音编码", "Soundex", "紧凑的英语语音键。"),
    "metaphone": ("语音编码", "Metaphone", "英语语音编码，支持 Philips 与 Jellyfish 规则变体。"),
    "double-metaphone": ("语音编码", "Double Metaphone", "返回主要与备用读音键。"),
    "nysiis": ("语音编码", "NYSIIS", "纽约州身份识别与情报系统编码。"),
    "match-rating": ("语音编码", "Match Rating 方法", "生成编码并直接比较姓名。"),
    "caverphone": ("语音编码", "Caverphone 2.0", "为新西兰记录设计的英语姓名编码。"),
    "cologne": ("语音编码", "科隆语音算法", "支持统一码的德语语音编码。"),
    "daitch-mokotoff": (
        "语音编码",
        "Daitch–Mokotoff Soundex",
        "面向斯拉夫和意第绪语姓氏、可产生分支的六位数字编码。",
    ),
    "beider-morse": ("语音编码", "Beider–Morse", "用于族谱姓名的多语言语音匹配。"),
}


ZH_VECTORIZATION_TEXT = {
    "shape-one-to-many": (
        "单个查询对多个目标",
        "1 个查询 + T 个目标",
        "T 个分数",
        "按目标原有顺序为每个目标返回一个分数。一个查询需要与一组候选项比较时，应直接使用这种批量形状。",
    ),
    "shape-ranking": (
        "最佳匹配与一对多前 k 项",
        "1 个查询 + T 个目标 + K",
        "1 个匹配或最多 K 个匹配",
        "最佳匹配只返回一个候选项；前 k 项仅保留指定数量的候选项，应用代码无需构造并排序完整分数向量。",
    ),
    "shape-cdist-matrix": (
        "稠密全配对矩阵",
        "Q 个查询 × T 个目标",
        "Q 行 × T 列",
        "cdist 为笛卡尔积评分。第 q 行对应第 q 个查询，第 t 列对应第 t 个目标，单元格 [q, t] 就是这一对的分数。",
    ),
    "shape-global-selection": (
        "阈值筛选与全局前 k 项",
        "概念上的 Q × T 网格",
        "达到阈值的配对记录，或全局最多 K 项",
        "阈值形式返回所有达标配对。全局前 k 项让整个矩阵共同竞争，因此 k=2 表示总共只返回两项，有些查询可能没有结果。",
    ),
    "shape-per-query-selection": (
        "每个查询各自的前 k 项",
        "Q 个查询 × T 个目标 + K",
        "Q 个分组，每组最多 K 个匹配",
        "每个查询都有独立的候选竞争。k=2 时最多可返回 2Q 个匹配，并且每个非空查询行都有自己的近邻列表。",
    ),
}


def _selector(slug: str, locale: str) -> str:
    select_id = f"api-language-{slug}"
    label = "语言" if locale == "zh-CN" else "Language"
    return f"""<div class="api-example-select">
<label for="{select_id}">{label}</label>
<span><select id="{select_id}" data-language-select>
<option value="python">Python</option>
<option value="r">R</option>
<option value="duckdb">DuckDB</option>
</select></span>
</div>"""


def _panel(language: str, code: str) -> str:
    imports = {
        "python": "import stride_align as sa\n\n",
        "r": "library(stridealign)\n\n",
        "duckdb": "",
    }
    prefix = imports[language]
    if code.startswith("from stride_align"):
        prefix = "import stride_align as sa\n"
    return (
        f'<div class="api-language-panel" data-language-panel="{language}"'
        + ("" if language == "python" else " hidden")
        + f"><pre><code>{escape(prefix + code)}</code></pre></div>"
    )


def render_api_catalog(locale: str = "en") -> str:
    cards = []
    for example in EXAMPLES:
        family, title, description = (
            ZH_EXAMPLE_TEXT[example.slug]
            if locale == "zh-CN"
            else (example.family, example.title, example.description)
        )
        detail_label = "详细语义（英文） →" if locale == "zh-CN" else "Detailed semantics →"
        detail = (
            f'<a class="api-detail-link" href="{escape(example.detail)}">{detail_label}</a>'
            if example.detail
            else ""
        )
        cards.append(
            f"""<section class="api-example" id="{escape(example.slug)}">
<header class="api-example-heading">
<div><span>{escape(family)}</span><h2>{escape(title)}</h2></div>
{_selector(example.slug, locale)}
</header>
<p>{escape(description)}</p>
<div class="api-language-panels">
{_panel("python", example.python)}
{_panel("r", example.r)}
{_panel("duckdb", example.duckdb)}
</div>
{detail}
</section>"""
        )
    return '<div class="api-catalog">\n' + "\n".join(cards) + "\n</div>"


def render_vectorization_guide(locale: str = "en") -> str:
    cards = []
    for example in VECTORIZATION_EXAMPLES:
        title, input_shape, output_shape, description = (
            ZH_VECTORIZATION_TEXT[example.slug]
            if locale == "zh-CN"
            else (
                example.title,
                example.input_shape,
                example.output_shape,
                example.description,
            )
        )
        shape_label = "向量化形状" if locale == "zh-CN" else "Vectorization shape"
        input_label = "输入" if locale == "zh-CN" else "Input"
        output_label = "输出" if locale == "zh-CN" else "Output"
        cards.append(
            f"""<section class="api-example vectorization-example" id="{escape(example.slug)}">
<header class="api-example-heading">
<div><span>{shape_label}</span><h2>{escape(title)}</h2></div>
{_selector(example.slug, locale)}
</header>
<div class="vector-shape" aria-label="Input and output shape">
<div><span>{input_label}</span><code>{escape(input_shape)}</code></div>
<b aria-hidden="true">→</b>
<div><span>{output_label}</span><code>{escape(output_shape)}</code></div>
</div>
<p>{escape(description)}</p>
<div class="api-language-panels">
{_panel("python", example.python)}
{_panel("r", example.r)}
{_panel("duckdb", example.duckdb)}
</div>
</section>"""
        )
    return '<div class="api-catalog vectorization-catalog">\n' + "\n".join(cards) + "\n</div>"
