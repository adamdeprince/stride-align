# stride-align

**語言:** [English](README.md) · [简体中文](README.zh-CN.md) · **[繁體中文](README.zh-TW.md)** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` 是一個[快得冒煙的函式庫](BENCHMARK.md)，用來告訴你兩個字串有多「相似」。它透過實作 Smith-Waterman 和 Needleman-Wunsch 演算法來做到這件事。這裡不講大課，我們直接邊做邊學。先看看它怎麼運作。

## 安裝

```bash
pip install stride-align
```

在 Loongson 系統上，先從你的 Linux 發行版安裝 NumPy，再安裝 `stride-align`：

```bash
sudo apt install python3-numpy
pip install stride-align
```

先說一句免責聲明：我在這裡使用宗教文本不是為了推動任何議程。這個示範需要幾份比較長、屬於公有領域、意思相同但措辭不同的文件。聖經剛好非常離譜地符合這個示範需求。

想像我們有兩個句子；這裡用《創世記》的第一句。

在 American Standard Version 中是："In the beginning God created the heavens and the earth."

在 King James Version 中是："In the beginning God created the heaven and the earth."

肉眼能看出差異：heavens 和 heaven。但我們要怎麼量化這個差異？可以用下面這小段程式：

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

執行後會印出：

```python
0.9907407407407407
```

正規化分數位於 `0` 到 `1` 之間。分數 `1` 代表在預設計分模型下輸入完全相符。接近 `0` 的分數代表輸入幾乎沒有共同點，不過 Smith-Waterman 仍然可能在彼此無關的字串中找到很小的局部相符片段。

現在換一段文字，看看分數會怎麼變。

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

然後 Python 會印出：

```
0.12222222222222222
```

開始有感覺了嗎？字串越相似，分數就越高。

接下來做一個更大的例子，讓我們對這個函式庫的效能有點直覺。你可能會注意到我們會在 Smith-Waterman 和 Needleman-Wunsch 之間切換，也可能會想什麼時候該用哪一個。想比較整個輸入和整個輸入時，用 Needleman-Wunsch。想在較大的輸入裡找出最相符的區域時，用 Smith-Waterman。

好，繼續看示範程式。這部分示範需要 `requests`：

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

現在要怎麼用它？假設我們有一段隨機的聖經經文，想知道它出自哪一章哪一節。你說 `grep`？唉，不行，我們犯了一個錯。手上的經文來自另一個譯本，例如 Catholic Public Domain，而電腦上只有 King James Bible。`grep` 的精確字串比對在這裡派不上用場。那要怎麼找到章節？當然是用 `stride-align` 搜尋「最接近」或「最相似」的字串。

在示範的第一部分，我們處理下載和快取。[Open Bible](https://openbible.com) 的好心人把文字放在可以透過 HTTP 取得的位置，但我們也要尊重他們的 IT 預算，所以會快取下載內容。這就是良好網路公民的基本禮貌。

下一部分把所有行載入列表。我們去掉換行，並把所有內容轉成小寫，因為不想為了是否按著 Shift 鍵這種事搞得太挑剔。

最後，`while True:` 迴圈收集一行文字；大概就是我們想從 Catholic 版本聖經裡查找章節的那句經文。然後它用批次形式的 Needleman-Wunsch，把這行文字和 King James Bible 的所有行比對。它會傳回一個分數陣列。我們用 `argmax()` 找出分數最高的行，再印出該索引對應的文字。試試看。

我會用 Catholic Bible 的 Jeremiah 4:28；它和 King James Bible 裡的同一節其實差很多。看看會發生什麼……

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

……找到了！而且還挺快。

## 再來一個示範：拼字檢查

這是一個玩具級拼字檢查器，不是正式產品。它忽略標點、大小寫、詞頻、專有名詞和上下文。重點是展示同樣的「一個查詢對許多候選項」模式，用在一個熟悉的任務上。

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

這個腳本做的第一件事，是嘗試找到作業系統中的正確拼字單字列表。位置可能因發行版而異。找到之後，我們載入它、去掉換行，然後開始拼字檢查。

拼字檢查和前面的比對很像。對每個候選單字，我們把它和正確拼字單字列表裡的所有單字比對，用 `argmax()` 找出分數最高的候選項，再用那個候選項替換原單字。我們當然可以用一些最佳化讓它更快，例如不要為已經拼對的單字搜尋匹配；不過這只是示範，這個最佳化就留給讀者練習。

看看它怎麼運作！

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## 細節

目前的腳手架提供：

- Needleman-Wunsch 只計算分數的比對
- 帶回溯的 Needleman-Wunsch 比對
- Smith-Waterman 只計算分數的比對
- 帶回溯的 Smith-Waterman 比對
- 一種後端布局，符合 `massive-speedup` 使用的專門化模式
- CPU/後端偵測，以及 Python 端的後端分派

原生邊界接受：

- `bytes` 對 `bytes`
- `str` 對 `str`
- 不可變、可雜湊的 Python 物件序列
- 混合的序列/物件輸入；其中 `str` 或 `bytes` 的一側會被視為序列處理

直接把 `bytes` 和 `str` 成對比較會拋出 `TypeError`。

目前實作是通用的動態規劃核心，帶有預處理步驟：把 Python 輸入序列化成 8、16、32 或 64 位元的標記流。後續可以用 SIMD 專門化後端替換後端翻譯單元，而不改變 Python API。

只計算分數的函式會傳回數值分數。正規化版本會傳回 `0` 到 `1` 之間的分數。路徑函式會傳回比對結果物件，內含分數、對齊後的值、操作序列，以及在可用時的 CIGAR 風格摘要。

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

對於「一個查詢對多個目標」的分數工作負載，使用 `Scores(...).compare([...])` 或 `*_scores()` 函式。這條路徑只準備一次查詢/設定檔，是反覆進行英文/中文文字比較時首選的高效能 API。

回溯輸出會保留配對的快速路徑型別：

- `str` 輸入傳回對齊後的 `str`
- `bytes` 輸入傳回對齊後的 `bytes`
- 序列/物件輸入傳回帶有 `None` 空位的對齊 `tuple` 值

傳入 `width=8`、`16`、`32` 或 `64`，可以強制內部標記/計分寬度，而不是使用自動選擇。

有些函式會暴露 CIGAR 字串，CIGAR 是 “Concise Idiosyncratic Gapped Alignment Report” 的縮寫。它是 SAM/BAM 工具使用的緊湊比對操作記法。如果你想看完整的正式版本，請參閱 [SAM 規範](https://samtools.github.io/hts-specs/SAMv1.pdf)。

## 最佳化與基準測試

我一直認真關注 `stride-align` 的效能表現，而且會持續如此。這個函式庫包含面向多種常見目標的 SIMD 最佳化，包括 x86、Arm 和 LoongArch。

**LoongArch / Loongson.** Loongson 的最佳化故事尤其有說服力：在已檢查的基準測試案例中，針對英文文字、16 位元分數寬度、只計算分數的 Smith-Waterman，LASX 後端比通用後端快 16 倍，比 Parasail 快 **22.4 倍**。

如果你是使用 Loongson 伺服器並受益於這個加速的研究人員，歡迎引用、提交 bug 報告、提供基準測試案例，以及寄一點便宜的小型中國紀念品。茶葉、書法書籤、剪紙飾品、中國結掛件、熊貓鑰匙圈和小龍桌面擺件都歡迎。請不要寄任何昂貴物品，也不要寄需要海關文件的東西。

參見[完整基準測試](BENCHMARKS.md)。

## 原生微基準測試

如果要在沒有 Python 堆疊框架或基準測試編排開銷的情況下做效能分析，可以設定一個原生 x86 微基準建置：

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` 會保留未剝離的 nanobind 模組，並加入除錯符號和框架指標，同時保留 `-O3`。

已提交的原生微基準基線位於 `benchmarks/x86_microbench_baseline.json`。把它當作帶有寬鬆閾值的本地護欄，不要當作跨機器 SLA。


## 引用

如果你在研究中使用我的軟體，請引用我。

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
