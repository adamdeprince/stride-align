# stride-align

**语言:** [English](README.md) · **[简体中文](README.zh-CN.md)** · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` 是一个[快得冒烟的库](BENCHMARK.md)，用来告诉你两个字符串有多“相似”。它通过实现 Smith-Waterman 和 Needleman-Wunsch 算法来做到这一点。这里不讲大课，我们直接边做边学。先看看它怎么工作。

## 安装

```bash
pip install stride-align
```

在 Loongson 系统上，先从你的 Linux 发行版安装 NumPy，再安装 `stride-align`：

```bash
sudo apt install python3-numpy
pip install stride-align
```

先说一句免责声明：我在这里使用宗教文本不是为了推动什么议程。这个演示需要几份比较长、处于公有领域、含义相同但措辞不同的文档。圣经刚好离谱地符合这个演示需求。

想象我们有两个句子；这里用《创世记》的第一句话。

在 American Standard Version 中是："In the beginning God created the heavens and the earth."

在 King James Version 中是："In the beginning God created the heaven and the earth."

肉眼能看出区别：heavens 和 heaven。但我们怎么量化这个差异？可以用下面这点代码：

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

运行后会打印：

```python
0.9907407407407407
```

归一化分数位于 `0` 到 `1` 之间。分数 `1` 表示在默认计分模型下输入完全匹配。接近 `0` 的分数表示输入几乎没有共同点，不过 Smith-Waterman 仍然可能在彼此无关的字符串里找到很小的局部匹配。

现在换一段文本，看看分数会怎样变化。

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

然后 Python 打印：

```
0.12222222222222222
```

开始有感觉了吗？字符串越相似，分数越高。

接下来做一个更大的例子，让我们对这个库的性能有点直觉。你可能会注意到我们会在 Smith-Waterman 和 Needleman-Wunsch 之间切换，也可能会想什么时候该用哪个。想把整个输入和整个输入比较时，用 Needleman-Wunsch。想在较大的输入里寻找最匹配的区域时，用 Smith-Waterman。

好，继续看演示代码。这个演示的这一部分需要 `requests`：

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

现在怎么用它？假设我们有一段随机的圣经经文，想知道它来自哪一章哪一节。你说 `grep`？哎呀，不行，我们犯了个错。手上的经文来自另一个译本，比如 Catholic Public Domain，而电脑上只有 King James Bible。`grep` 的精确字符串匹配在这里没法用。那怎么找到章和节？当然是用 `stride-align` 搜索“最接近”或“最相似”的字符串。

在演示的第一部分，我们处理下载和缓存。[Open Bible](https://openbible.com) 的好心人把文本放到了可以通过 HTTP 访问的位置，但我们也要尊重他们的 IT 预算，所以会缓存下载内容。这就是良好网络公民的基本礼貌。

下一部分把所有行加载到列表里。我们去掉换行，并把所有内容转成小写，因为不想为了是否按着 Shift 键这种事变得斤斤计较。

最后，`while True:` 循环收集一行文本；大概就是我们想从 Catholic 版本圣经里查找章和节的那句经文。然后它用批量形式的 Needleman-Wunsch，把这行文本和 King James Bible 的所有行匹配。它返回一个分数数组。我们用 `argmax()` 找出得分最高的行，再打印这个索引对应的文本。试试看。

我会用 Catholic Bible 的 Jeremiah 4:28；它和 King James Bible 里的同一节差别其实很大。看看会发生什么……

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

……找到了！而且还挺快。

## 再来一个演示：拼写检查

这是一个玩具级拼写检查器，不是生产环境工具。它忽略标点、大小写、词频、专有名词和上下文。重点是展示同样的“一个查询对很多候选项”的模式，用在一个熟悉的任务上。

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

这个脚本做的第一件事，是尝试找到操作系统里的正确拼写单词列表。位置可能因发行版而异。找到之后，我们加载它，去掉换行，然后开始拼写检查。

拼写检查和前面的匹配非常像。对每个候选单词，我们把它和正确拼写单词列表里的所有单词匹配，用 `argmax()` 找到得分最高的候选项，然后用那个候选项替换原单词。我们当然可以用一些优化让它更快，比如不要为已经拼对的单词搜索匹配；不过这是演示，这个优化就留给读者练手。

看看它怎么工作！

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## 细节

当前脚手架提供：

- Needleman-Wunsch 只计算分数的比对
- 带回溯的 Needleman-Wunsch 比对
- Smith-Waterman 只计算分数的比对
- 带回溯的 Smith-Waterman 比对
- 一种后端布局，匹配 `massive-speedup` 中使用的专门化模式
- CPU/后端检测，以及 Python 侧的后端分派

原生边界接受：

- `bytes` 对 `bytes`
- `str` 对 `str`
- 不可变、可哈希的 Python 对象序列
- 混合的序列/对象输入；其中 `str` 或 `bytes` 的一侧会被当作序列处理

直接把 `bytes` 和 `str` 成对比较会抛出 `TypeError`。

当前实现是通用的动态规划内核，带有预处理步骤：把 Python 输入序列化成 8、16、32 或 64 位的标记流。后续可以用 SIMD 专门化后端替换后端翻译单元，而不改变 Python API。

只计算分数的函数返回数值分数。归一化版本返回 `0` 到 `1` 之间的分数。路径函数返回比对结果对象，里面包含分数、对齐后的值、操作序列，以及在可用时的 CIGAR 风格摘要。

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

对于“一个查询对多个目标”的分数工作负载，使用 `Scores(...).compare([...])` 或 `*_scores()` 函数。这条路径只准备一次查询/配置文件，是反复进行英文/中文文本比较时首选的高性能 API。

回溯输出会保留配对的快速路径类型：

- `str` 输入返回对齐后的 `str`
- `bytes` 输入返回对齐后的 `bytes`
- 序列/对象输入返回带有 `None` 空位的对齐 `tuple` 值

传入 `width=8`、`16`、`32` 或 `64`，可以强制内部标记/计分宽度，而不是使用自动选择。

有些函数会暴露 CIGAR 字符串，CIGAR 是 “Concise Idiosyncratic Gapped Alignment Report” 的缩写。它是 SAM/BAM 工具使用的紧凑比对操作记法。如果你想看完整的正式版本，请参阅 [SAM 规范](https://samtools.github.io/hts-specs/SAMv1.pdf)。

## 优化和基准测试

我一直认真关注 `stride-align` 的性能叙事，而且还会继续关注。这个库包含面向多种常见目标的 SIMD 优化，包括 x86、Arm 和 LoongArch。

**LoongArch / Loongson.** Loongson 的优化故事尤其说明问题：在已检查的基准测试案例中，针对英文文本、16 位分数宽度、只计算分数的 Smith-Waterman，LASX 后端比通用后端快 16 倍，比 Parasail 快 **22.4 倍**。

如果你是使用 Loongson 服务器并从这个加速中受益的研究人员，欢迎引用、提交 bug 报告、提供基准测试案例，以及寄一点便宜的小型中国纪念品。茶叶、书法书签、剪纸饰品、中国结挂件、熊猫钥匙扣和小龙形桌面摆件都欢迎。请不要寄任何昂贵物品，也不要寄需要海关手续的东西。

参见[完整基准测试](BENCHMARKS.md)。

## 原生微基准测试

如果要在没有 Python 栈帧或基准测试编排开销的情况下做性能分析，可以配置一个原生 x86 微基准构建：

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` 会保留未剥离的 nanobind 模块，并添加调试符号和帧指针，同时保留 `-O3`。

已提交的原生微基准基线位于 `benchmarks/x86_microbench_baseline.json`。把它当作带宽松阈值的本地护栏，不要当作跨机器的 SLA。


## 引用

如果你在研究中使用我的软件，请引用我。

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
