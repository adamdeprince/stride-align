# stride-align

**Languages:** [English](README.md) · [简体中文](README.zh-CN.md)

`stride-align` 是一个[极速库](https://stride-align.com/BENCHMARK.html)，用来判断两个字符串的
“相似程度”。它通过实现 Smith-Waterman 和 Needleman-Wunsch 算法来完成
这件事。这里不讲理论，直接动手——边做边学。

## 安装

```bash
pip install stride-align
```

在龙芯（Loongson）系统上，请先从你的 Linux 发行版安装 NumPy，再
安装 `stride-align`；龙架构（LoongArch64）的 wheel 需要从下面任一
镜像获取 —— PyPI 暂不接受 `linux_loongarch64` 或
`manylinux_2_38_loongarch64` 平台标签，所以二进制安装目前只有这
一条路径。

```bash
sudo apt install python3-numpy
PY=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
```

从 GitHub 安装：

```bash
pip install \
  https://github.com/adamdeprince/stride-align/releases/download/v0.3.0/stride_align-0.3.0-${PY}-${PY}-linux_loongarch64.whl
```

如果从 GitHub 下载不方便，可以改用 `stride-align.com` 上的镜像；
这里提供的是同一份 wheel：

```bash
pip install \
  https://stride-align.com/wheels/v0.3.0/stride_align-0.3.0-${PY}-${PY}-linux_loongarch64.whl
```

预编译的龙架构（LoongArch64）wheel 在两个镜像上都覆盖 Python
3.12、3.13 和 3.14。如果你用的是其他 Python 版本
（或者想从源码构建），`pip install stride-align` 会回退到 PyPI
上的源码发行版，在本地编译 LSX/LASX 内核。

先声明一下：这里用宗教文本不是要带任何立场——这个 demo 需要几份
比较大的、含义相同但表达不同的公共领域文档，圣经恰好极其符合这个
要求。

假设我们有两个句子——用《创世记》的开篇：

美国标准译本（American Standard Version，ASV）是这样的：“In the
beginning God created the heavens and the earth.”

英王钦定版（King James Version，KJV）则是：“In the beginning God created
the heaven and the earth.”

肉眼就能看到区别——heavens 还是 heaven。但怎么把这种差异量化？用
下面这一小段代码：

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

运行后输出：

```python
0.9907407407407407
```

归一化得分在 `0` 和 `1` 之间。得分为 `1` 表示在默认评分模型下两个
输入完全匹配。得分接近 `0` 表示两份输入几乎没有共同之处，不过
Smith-Waterman 仍可能在原本不相关的字符串内部找到小的局部匹配。

现在换一下文本，看看得分怎么变。

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

Python 输出：

```
0.12222222222222222
```

明白了吗？字符串越相似，得分越高。

我们来做个稍大点的例子，感受一下这个库的性能。你可能会注意到我们
在 Smith-Waterman 和 Needleman-Wunsch 之间切换，大概会想什么时候
用哪一个。当你想拿整段输入与整段输入比较时，用 Needleman-Wunsch。
当你想在较大的输入内部找出匹配最好的区域时，用 Smith-Waterman。

好，进入 demo 代码。这一部分需要 `requests`：

```bash
pip install requests
```

```python
import os, time, requests
import stride_align as sa

if not os.path.exists("kjv.txt"):
    response = requests.get("https://openbible.com/textfiles/kjv.txt")
    response.raise_for_status()
    response.encoding = "utf-8-sig"
    open("kjv.txt", "w", encoding="utf-8").write(response.text)

lines = [line.strip().lower() for line in open("kjv.txt")][2:]

while True:
    if not (query := input("Enter a snippet to match.  Press enter to end.\n")):
        break
    t = time.perf_counter()
    scores = sa.needleman_wunsch_normalized_scores(query.lower(), lines)
    best = int(scores.argmax())
    print()
    print("Score:", float(scores[best]))
    print(lines[best])
    print("Search time: %0.2fms" % ((time.perf_counter() - t) * 1000))
    print()
    print()
```

这能拿来干什么？假设我们手上有一段圣经经文，想知道它出自哪一章
哪一节。你说用 `grep`？哎，不行，这里有个问题：我们手上的经文是
另一个译本，比方说 Catholic Public Domain 译本，而电脑上的是
King James 版本。`grep` 的精确匹配在这里行不通。怎么找到对应的
章节？当然是用 `stride-align` 搜索“最接近”或“最相似”的字符串。

demo 的第一部分处理下载和缓存。[Open Bible](https://openbible.com)
把这份文本放在了 HTTP 可访问的位置，我们要尊重他们的带宽，所以
下载下来就缓存住，做个体面的网络公民。

接下来把所有行加载到一个列表里。我们去掉换行符，把所有内容转成
小写——免得在 Shift 键问题上较劲。

最后那段 `while True:` 循环读取一行文本——大概就是来自 Catholic
版本、我们想查章节的那句经文——用 Needleman-Wunsch 的批处理形式
把它和 King James 版本里所有行做匹配。它返回一个得分数组。我们
用 `argmax()` 找到最高分对应的行，再把对应索引的那行打印出来。
试一下。

我准备用 Catholic 圣经里的 Jeremiah 4:28——它和 King James 圣经
里的同一节其实差别挺大。来看看效果……

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

……找到了！而且相当快。

再来一个 demo：拼写检查。

这是一个玩具拼写检查器，不是生产级的。它忽略标点、大小写、词频、
专有名词和上下文。重点是用一个熟悉的任务展示同样的“一对多查询”
模式。

```python
import os, sys
import stride_align as sa

paths = ['/usr/share/dict/words',
         '/usr/dict/words',
         '/var/lib/dict/words',
         '/etc/dictionaries-common/words']

for path in paths:
    if os.path.exists(path):
        break
else:
    print("Sorry, I can't find your dictionary", file=sys.stderr)
    exit(1)


words = [line.strip().lower() for line in open(path)]


for line in sys.stdin:
    new_line = []
    for word in line.split():
        scores = sa.needleman_wunsch_normalized_scores(word.lower(), words)
        word = words[int(scores.argmax())]
        new_line.append(word)
    print(' '.join(new_line), flush=True)
```

脚本做的第一件事是尝试找到操作系统里的拼写正确词表。它的位置因
发行版而异。找到之后，加载、去掉换行符，然后开始拼写检查。

拼写检查本身跟前面的匹配很像。对每个输入词，把它和正确词表里的
所有词做匹配，用 `argmax()` 找到分最高的那个，然后用它替换。
这里还有不少可以优化的地方，比如对已经拼对的词直接跳过查找，
不过这是个 demo，这点优化就留作读者练习了。

来看看效果！

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## 细节

当前实现提供：

- Needleman-Wunsch 仅得分对齐
- Needleman-Wunsch 带回溯的对齐
- Smith-Waterman 仅得分对齐
- Smith-Waterman 带回溯的对齐
- 与 `massive-speedup` 中专门化模式一致的 backend 布局
- CPU/backend 检测，以及 Python 层的 backend dispatch

C++/Python 边界接受：

- `bytes` 对 `bytes`
- `str` 对 `str`
- 由不可变可哈希 Python 对象组成的序列
- 混合的“序列/对象”输入——某一侧的 `str` 或 `bytes` 会被当作
  序列处理

直接拿 `bytes` 对 `str` 配对会抛出 `TypeError`。

当前实现是通用的动态规划内核，配上把 Python 输入序列化成 8、16、
32 或 64 位 token 流的预处理。SIMD 专门化的 backend 之后可以替换
对应的 backend 编译单元，而不动 Python API。

仅得分函数返回数值得分。归一化版本返回 `0` 到 `1` 之间的分数。
路径函数返回对齐结果对象，里面包含得分、对齐后的值、操作序列，
以及可用时的 CIGAR 摘要。

## API

```python
import stride_align

score = stride_align.needleman_wunsch_score("ACGT", "ACCT")
scores = stride_align.Scores("ACGT", variant="needleman_wunsch").compare(["ACCT", "AGGT"])
result = stride_align.smith_waterman_path("ACCGT", "CCG")
wide_result = stride_align.smith_waterman_path("ACCGT", "CCG", width=64)
object_result = stride_align.needleman_wunsch_path(
    [frozenset({1}), frozenset({2})],
    [frozenset({1}), frozenset({3})],
)

print(score)
print(scores)
print(result.score, result.aligned_query, result.aligned_target, result.operations)
print(wide_result.score)
print(object_result.aligned_query, object_result.aligned_target)
```

“一个 query 对多个 target”的得分工作流，请使用
`Scores(...).compare([...])` 或者 `*_scores()` 函数。这条路径
会一次性准备好 query / profile，是反复做中英文文本比对时推荐的
高性能 API。

对于直接配对的快速路径，回溯输出会保留输入类型：

- `str` 输入返回对齐后的 `str`
- `bytes` 输入返回对齐后的 `bytes`
- 序列/对象输入返回对齐后的 `tuple` 值，空位用 `None`

传 `width=8`、`16`、`32` 或 `64` 可以强制内部 token / 评分宽度，
而不是用自动选择。

部分函数会暴露 CIGAR 字符串——CIGAR 是 “Concise Idiosyncratic
Gapped Alignment Report” 的缩写，是 SAM/BAM 工具链使用的紧凑
对齐操作记法。如果想看完整规范，见
[SAM specification](https://samtools.github.io/hts-specs/SAMv1.pdf)。

### 替换矩阵（BLOSUM、PAM）

针对蛋白质比对，`stride_align.matrices` 提供了规范的 BLOSUM
与 PAM 替换矩阵。可通过 `smith_waterman_score`、
`needleman_wunsch_score` 及其 `_scores` 批量版本上的 `matrix=`
关键字传入：

```python
import stride_align
from stride_align.matrices import blosum62, pam250

# 局部比对，NCBI 标准 BLOSUM62 + 仿射空位（open=-11, extend=-1）。
# matrix= 与 match_score / mismatch_score 互斥。
stride_align.smith_waterman_score(
    "HEAGAWGHEE", "PAWHEAE",
    matrix=blosum62,
    gap_open_score=-11, gap_extend_score=-1,
)

# 批量（1 条查询 × N 条目标），共用一次构建好的剖面 —— 推荐用法。
stride_align.smith_waterman_scores(
    "HEAGAWGHEE",
    ["PAWHEAE", "HEAGAWGHEE", "MEEPS"],
    matrix=pam250, gap_open_score=-14, gap_extend_score=-2,
)

# 自定义矩阵：直接解析 NCBI 文本格式
custom = stride_align.matrices.SubstitutionMatrix.from_ncbi_text(
    open("/path/to/BLOSUM62").read(),
    name="BLOSUM62",
    gap_open=-11, gap_extend=-1,
)
```

每个内置 `SubstitutionMatrix` 都暴露字母表、矩阵数据
（`int8` ndarray）以及推荐的空位默认值 `.gap_score`（线性）、
`.gap_open`、`.gap_extend`。线性空位（`gap_score=`）与仿射空位
（`gap_open_score=` + `gap_extend_score=`）在 AVX-512 后端都受支持；
其它 SIMD 后端目前对矩阵模式会回退到标量通用内核。

矩阵数值来自 NCBI BLAST 发行版
[`ftp.ncbi.nih.gov/blast/matrices/`](https://ftp.ncbi.nih.gov/blast/matrices/)，
即规范的参考分值。原始文献为：

- **BLOSUM45 / 50 / 62 / 80 / 90** —— Henikoff S.、Henikoff J.G.（1992）。
  *Amino acid substitution matrices from protein blocks*。PNAS
  89(22):10915–10919。
  [doi:10.1073/pnas.89.22.10915](https://doi.org/10.1073/pnas.89.22.10915)
  &nbsp;·&nbsp;
  [PDF（开放获取）](https://www.pnas.org/doi/pdf/10.1073/pnas.89.22.10915)
- **PAM30 / 70 / 250** —— Dayhoff M.O.、Schwartz R.M.、Orcutt B.C.
  （1978）。*A model of evolutionary change in proteins*。收于
  *Atlas of Protein Sequence and Structure*，第 5 卷，补遗 3，
  第 345–352 页。美国国家生物医学研究基金会（NBRF），华盛顿特区。
  （书籍章节，无开放 PDF。常引用的后续推导见
  Schwartz R.M.、Dayhoff M.O.（1978），*Matrices for detecting distant
  relationships*，同卷第 353–358 页。）

### 编辑距离评分器

除了 Smith-Waterman 和 Needleman-Wunsch，`stride-align` 还提供六种
单位代价的编辑距离/相似度度量。其中大多数都有 SIMD 批处理代码
路径；true-DL 目前是例外，仍然走标量 DP：

```python
import stride_align

# Levenshtein（Myers 1999 位并行）——插入、删除、替换
stride_align.levenshtein_score("kitten", "sitting")               # -> 3
stride_align.levenshtein_normalized_score("kitten", "sitting")    # -> 0.571...
stride_align.levenshtein_scores("kitten", ["kit", "sitting"])     # -> ndarray[int64]

# 可选的 `score_cutoff`（rapidfuzz 的约定）：按 target 提前退出，
# 超过 cutoff 的结果以 `cutoff + 1` 形式返回。
stride_align.levenshtein_scores(query, targets, score_cutoff=3)

# Damerau-Levenshtein（OSA 受限版，Hyyrö 2002）——在单位代价
# 下增加相邻字符的转置。这是 rapidfuzz 用 OSA.distance 暴露的
# 那个，也是大多数人嘴里说的“Damerau-Levenshtein”实际上想要的。
stride_align.damerau_levenshtein_score("ab", "ba")                # -> 1

# True Damerau-Levenshtein ——不受限版本，一个字符可以参与多次
# 编辑。较慢（目前没有位并行内核），但与
# rapidfuzz.distance.DamerauLevenshtein 完全一致。和 OSA 在重叠
# 转置上会分叉，例如 "ca" -> "abc"：OSA=3，true-DL=2。
stride_align.true_damerau_levenshtein_score("ca", "abc")          # -> 2

# Indel ——仅限插入和删除的 Levenshtein，不允许替换。
# 等价于 |a| + |b| - 2 * LCS(a, b)。Allison-Dix（1986）位并行
# 内层循环。
stride_align.indel_score("kitten", "sitting")                     # -> 5

# Hamming ——两个等长字符串中位置不同的个数。
# 带 cutoff 的变体在累计差异超出上限时直接退出。
stride_align.hamming_score("100", "110")                          # -> 1

# Jaro / Jaro-Winkler ——[0, 1] 区间的相似度；Winkler 在前缀
# 上加一个有上限的加成。
stride_align.jaro_similarity("martha", "marhta")                  # -> 0.944...
stride_align.jaro_winkler_similarity("martha", "marhta")          # -> 0.961...
```

批处理变体（`*_scores`、`*_similarities`）在每一个支持的 backend
上都按“一个 target 一个 SIMD lane”的方式打包起来：

- x86：SSE4.1 / AVX2 / AVX-512 / AVX10-256 / AVX10-512
- ARM：NEON（Linux + macOS）、SVE / SVE2
- 龙架构（LoongArch）：LSX / LASX
- PowerPC：VSX

对 Lev / OSA 而言，长度 ≤ 64 的模式走单字 Myers；65–256 走多字
内核（W=2/3/4）。Indel 在模式长度 > 64 时回退到标量位并行
（多字推广暂未实现）；true-DL 目前只有标量 DP。

### `cdist`、`cdist_above_threshold`、`cdist_top_k`

针对两份字符串列表的全配对评分场景，`stride-align` 提供三个
矩阵风格的入口：

```python
qs = ["kitten", "sitting", "kit"]
ts = ["kitten", "kit", "sitting", "biting"]

# 完整 N×M 相似度矩阵 —— ndarray[float64]（相似度评分器）
# 或 ndarray[int64]（距离评分器）。
sa.cdist(qs, ts, scorer=sa.Scorer.JARO)

# 流式过滤 ——只产出相似度超过阈值的对。worker 向一个有界队列喂数据，
# 调用方负责消费。长度剪枝 + 把每对的 cutoff 下推到 kernel 里，
# 在高阈值下能跳过绝大部分工作。
for score, q, t in sa.cdist_above_threshold(
    qs, ts, scorer=sa.Scorer.LEVENSHTEIN_NORMALIZED, threshold=0.7,
):
    ...

# 按得分取 top-k ——返回最多 k 个最高得分（对距离评分器则是
# 最低）的 (score, query, target) 三元组。每个线程一个堆；
# 一个共享的原子全局最小值边界，让每对的 cutoff 下推
# 随着工作推进抬高剪枝阈值。
sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=10)
```

在高阈值下剪枝的效果非常夸张——见 [BENCHMARK.md](https://stride-align.com/BENCHMARK.html)
中的跨架构表（`cdist pruning` 那几行）。特别是龙芯（Loongson）的
LASX 后端在 T=0.99 下，让通常预期会领先的 Tiger Lake AVX-512
反而落后；对比报告见
[docs/loongson-vs-tiger-lake-cdist-2026-05-24.md](docs/loongson-vs-tiger-lake-cdist-2026-05-24.md)。

完整的跨架构数据见 [BENCHMARK.md](https://stride-align.com/BENCHMARK.html)。

## 优化和基准测试

`stride-align` 的性能一直被认真对待，而且仍在持续优化。这个库为
x86、Arm、龙架构（LoongArch）等多个常见目标做了 SIMD 优化。

**龙架构（LoongArch）/ 龙芯（Loongson）。** 龙芯的优化故事尤其
值得一说：在基准用例下（英文文本、16 位得分宽度、仅得分的
Smith-Waterman），LASX backend 比 generic backend 快 16 倍，
比 Parasail 快 **22.4 倍**。

如果你是在龙芯服务器上做研究，并且从这份提速里得到好处的研究者，
欢迎引用、提交 bug、贡献基准用例；如果你愿意表达一点心意，我也很喜欢收到一些不贵的中国小纪念品——
茶叶、书法书签、剪纸、中国结、熊猫钥匙扣、小龙摆件之类都很好。
请不要寄贵重物品或需要报关的东西。

完整基准测试见 [BENCHMARK.md](https://stride-align.com/BENCHMARK.html)。

## 原生微基准

如果想在不经 Python 帧、不走基准编排的情况下做性能分析，可以配置
一份原生 x86 microbench 构建：

```bash
nanobind_dir="$(.venv/bin/python -m nanobind --cmake_dir)"
cmake -S . -B build/perf \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSTRIDE_ALIGN_BUILD_MICROBENCH=ON \
  -DSTRIDE_ALIGN_PERF_SYMBOLS=ON \
  -DPython_EXECUTABLE=.venv/bin/python \
  -Dnanobind_DIR="$nanobind_dir"
cmake --build build/perf --target stride_align_x86_microbench
build/perf/stride_align_x86_microbench --backend avx2 --shape 1:many --pass english --width 16
python tools/x86_microbench_regression.py \
  --binary build/perf/stride_align_x86_microbench \
  --cpu 2 \
  --backends avx2,avx512bwvl \
  --shapes 1:1,1:many \
  --passes english,chinese \
  --widths 16,32 \
  --write-json /tmp/stride-align-x86-microbench.json
.venv/bin/python tools/pinned_benchmark_sweep.py \
  --output-dir /tmp/stride-align-pinned \
  --cpu 2 \
  --iterations 15 \
  --warmups 3
```

`STRIDE_ALIGN_PERF_SYMBOLS=ON` 会保留 nanobind 模块不被 strip，
并加入调试符号和 frame pointer，同时保持 `-O3`。

签入仓库的原生 microbench 基线在
`benchmarks/x86_microbench_baseline.json`。把它当作一个宽松阈值的
本地兜底，不要当成跨机器的 SLA。


## 引用

如果你在研究中用到了我的软件，请引用我。

```bibtex
@software{deprince_stride_align,
  author       = {DePrince, Adam},
  title        = {stride-align: Fast Smith-Waterman and Needleman-Wunsch alignment for Python},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/adamdeprince/stride-align},
  note         = {Python/C++ library for sequence and string alignment}
}
```
