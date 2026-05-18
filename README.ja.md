# stride-align

**言語:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · **[日本語](README.ja.md)** · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` は、2つの文字列がどれくらい「似ているか」を教えてくれる[爆速ライブラリ](BENCHMARK.md)です。Smith-Waterman と Needleman-Wunsch アルゴリズムを実装してそれを行います。講義は抜きにして、手を動かしながら覚えましょう。さっそく仕組みを見ていきます。

## インストール

```bash
pip install stride-align
```

Loongson システムでは、`stride-align` を入れる前に Linux ディストリビューションの NumPy をインストールしてください。

```bash
sudo apt install python3-numpy
pip install stride-align
```

最初に断っておくと、ここで宗教文書を使うのは何かの主張を押し出すためではありません。このデモには、同じ意味を持ちながら言い回しが違う、そこそこ大きいパブリックドメイン文書が複数必要です。聖書は、その条件に妙にぴったり合っているだけです。

2つの文があるとしましょう。ここでは創世記の最初の文を使います。

American Standard Version ではこうです: "In the beginning God created the heavens and the earth."

King James Version ではこうです: "In the beginning God created the heaven and the earth."

目で見ると違いは分かります。heavens と heaven です。では、この違いをどう数値化するのでしょうか。次の短いコードを使います。

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

実行すると次のように表示されます。

```python
0.9907407407407407
```

正規化スコアは `0` から `1` の間です。`1` は、デフォルトのスコアリングモデルで入力が完全一致したことを意味します。`0` に近いスコアは入力同士にほとんど共通点がないことを示しますが、Smith-Waterman は、ほぼ無関係な文字列の中にも小さな局所一致を見つけることがあります。

今度はテキストを変えて、スコアがどう変わるか見てみましょう。

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

すると Python は次を表示します。

```
0.12222222222222222
```

感覚がつかめてきましたか。文字列が似ているほど、スコアは高くなります。

次はもう少し大きい例を作って、このライブラリの性能を感じてみます。Smith-Waterman と Needleman-Wunsch を切り替えていることに気づいて、「いつどちらを使うのか」と思うかもしれません。入力全体と入力全体を比較したいときは Needleman-Wunsch を使います。大きな入力の中から最もよく一致する領域を探したいときは Smith-Waterman を使います。

ではデモコードへ進みます。この部分では `requests` が必要です。

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

どう使うのでしょうか。ランダムな聖書の一節があり、それが何章何節なのか知りたいとします。`grep` でいい？ いやいや、今回はミスがあります。その節は Catholic Public Domain のような別の翻訳から来ていて、手元のコンピュータには King James Bible しかありません。`grep` の完全一致検索ではうまくいきません。章と節をどう見つけるか。もちろん `stride-align` で「最も近い」または「最も似ている」文字列を検索します。

デモの最初の部分は、ダウンロードとキャッシュを扱います。[Open Bible](https://openbible.com) の親切な人たちは、このテキストを HTTP で取得できる場所に置いてくれています。ただし相手の IT 予算には敬意を払いたいので、ダウンロードしたものはキャッシュします。良い市民でいるための基本です。

次の部分では、すべての行をリストへ読み込みます。改行を取り除き、すべて小文字にします。Shift キーを押したかどうかで細かく騒ぎたくないからです。

最後に、`while True:` ループが1行のテキストを受け取ります。おそらく、章と節を調べたい Catholic 版聖書の節です。それを Needleman-Wunsch のバッチ形式で King James Bible の全行と照合します。返ってくるのはスコアの配列です。`argmax()` で最高スコアの行を見つけ、そのインデックスに対応する行を表示します。試してみましょう。

ここでは Catholic Bible の Jeremiah 4:28 を使います。King James Bible の同じ節とはかなり違います。何が起きるか見てみましょう。

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

……見つかりました。しかもけっこう速い。

## もう一つのデモ: スペルチェック

これはおもちゃのスペルチェッカーで、本番用ではありません。句読点、大文字小文字、単語頻度、固有名詞、文脈を無視します。目的は、同じ「1つの問い合わせを多数の候補に照合する」パターンを、なじみのあるタスクで見せることです。

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

このスクリプトが最初に行うのは、OS が持つ正しい綴りの単語リストを探すことです。場所はディストリビューションごとに違うことがあります。見つかったら読み込み、改行を取り除き、スペルチェックを始めます。

スペルチェックは前に行った照合とよく似ています。各候補単語について、正しい綴りの単語リストにあるすべての単語と照合し、`argmax()` で最高スコアの候補を探し、その候補で単語を置き換えます。正しく綴られている単語は検索しない、などの最適化で速くできますが、これはデモなので、その最適化は読者への練習問題として残しておきます。

動きを見てみましょう。

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## 詳細

現在の足場コードが提供するものは次のとおりです。

- Needleman-Wunsch のスコアのみのアラインメント
- トレースバック付き Needleman-Wunsch アラインメント
- Smith-Waterman のスコアのみのアラインメント
- トレースバック付き Smith-Waterman アラインメント
- `massive-speedup` で使っている特殊化パターンに合うバックエンド配置
- CPU/バックエンド検出と、Python 側のバックエンド分配

ネイティブ境界が受け付けるものは次のとおりです。

- `bytes` 対 `bytes`
- `str` 対 `str`
- 不変でハッシュ可能な Python オブジェクトのシーケンス
- 混在したシーケンス/オブジェクト入力。`str` または `bytes` 側はシーケンスとして扱われます

`bytes` と `str` の直接ペアは `TypeError` を送出します。

現在の実装は、前処理付きの汎用動的計画法カーネルです。前処理では Python 入力を 8、16、32、または 64 ビットのトークン列に直列化します。SIMD に特殊化したバックエンドは、Python API を変えずに後からバックエンド翻訳単位を置き換えられます。

スコアのみの関数は数値スコアを返します。正規化版は `0` から `1` のスコアを返します。パス関数はアラインメント結果オブジェクトを返し、そこにはスコア、整列済みの値、操作列、利用可能な場合は CIGAR 形式の要約が入ります。

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

1つの問い合わせを多数の対象に照合するスコア計算では、`Scores(...).compare([...])` または `*_scores()` 関数を使います。この経路は問い合わせ/プロファイルを一度だけ準備するため、英語/中国語テキストを繰り返し比較する高性能 API として推奨されます。

トレースバック出力は、対応する高速経路の型を保持します。

- `str` 入力は整列済みの `str` を返します
- `bytes` 入力は整列済みの `bytes` を返します
- シーケンス/オブジェクト入力は、`None` のギャップを含む整列済み `tuple` 値を返します

`width=8`、`16`、`32`、または `64` を渡すと、自動選択ではなく内部のトークン/スコア幅を強制できます。

一部の関数は CIGAR 文字列を公開します。CIGAR は “Concise Idiosyncratic Gapped Alignment Report” の略です。SAM/BAM ツールで使われるコンパクトなアラインメント操作表記です。正式な全体版が必要なら、[SAM 仕様](https://samtools.github.io/hts-specs/SAMv1.pdf) を参照してください。

## 最適化とベンチマーク

`stride-align` の性能面には、これまでも今後もかなり注意を払っています。このライブラリには、x86、Arm、LoongArch など一般的な対象向けの SIMD 最適化が含まれています。

**LoongArch / Loongson.** Loongson 向け最適化は特に分かりやすい例です。確認済みベンチマークのうち、英語テキスト、16 ビットのスコア幅、スコアのみを計算する Smith-Waterman というケースでは、LASX バックエンドは汎用バックエンドより 16 倍速く、Parasail より **22.4 倍** 速くなっています。

Loongson サーバを使ってこの高速化の恩恵を受けている研究者の方は、引用、バグ報告、ベンチマークケース、そして小さくて安価な中国土産を歓迎します。お茶、書道しおり、切り絵飾り、中国結び、パンダのキーホルダー、小さな龍の机上オブジェクトなどは大歓迎です。高価なものや税関書類が必要なものは送らないでください。

[完全なベンチマーク](BENCHMARKS.md) を参照してください。

## ネイティブ・マイクロベンチ

Python フレームやベンチマーク編成のオーバーヘッドなしで性能プロファイリングを行うには、ネイティブ x86 マイクロベンチビルドを構成します。

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` は nanobind モジュールをストリップせずに残し、デバッグシンボルとフレームポインタを追加しつつ、`-O3` を維持します。

リポジトリに入っているネイティブ・マイクロベンチのベースラインは `benchmarks/x86_microbench_baseline.json` にあります。これは緩いしきい値を持つローカルなガードレールとして扱い、マシンをまたぐ SLA として扱わないでください。


## 引用

研究で私のソフトウェアを使う場合は、引用してください。

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
