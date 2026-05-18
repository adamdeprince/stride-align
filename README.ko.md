# stride-align

**언어:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **[한국어](README.ko.md)** · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` 는 두 문자열이 얼마나 “비슷한지” 알려 주는 [엄청나게 빠른 라이브러리](BENCHMARK.md)입니다. Smith-Waterman 및 Needleman-Wunsch 알고리즘을 구현해 그 일을 합니다. 강의는 생략하고 직접 해 보면서 배우겠습니다. 바로 동작 방식을 보겠습니다.

## 설치

```bash
pip install stride-align
```

Loongson 시스템에서는 `stride-align`을 설치하기 전에 Linux 배포판의 NumPy를 먼저 설치하세요.

```bash
sudo apt install python3-numpy
pip install stride-align
```

먼저 짧게 밝히자면, 여기서 종교 문서를 쓰는 것은 어떤 의제를 밀기 위해서가 아닙니다. 이 데모에는 의미는 같지만 표현이 다른, 꽤 큰 공개 도메인 문서가 여러 개 필요합니다. 성경이 그 요구 조건에 이상할 정도로 잘 맞을 뿐입니다.

두 문장이 있다고 상상해 봅시다. 여기서는 창세기의 첫 문장을 쓰겠습니다.

American Standard Version 에서는 이렇게 되어 있습니다: "In the beginning God created the heavens and the earth."

King James Version 에서는 이렇게 되어 있습니다: "In the beginning God created the heaven and the earth."

눈으로 보면 차이가 보입니다. heavens 와 heaven 입니다. 하지만 이 차이를 어떻게 수치화할까요? 다음 작은 코드 조각을 씁니다.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

실행하면 다음이 출력됩니다.

```python
0.9907407407407407
```

정규화 점수는 `0`과 `1` 사이입니다. `1`은 기본 채점 모델에서 입력이 정확히 일치한다는 뜻입니다. `0`에 가까운 점수는 입력끼리 공통점이 거의 없다는 뜻이지만, Smith-Waterman 은 서로 무관해 보이는 문자열 안에서도 작은 국소 일치를 찾을 수 있습니다.

이제 텍스트를 바꾸고 점수가 어떻게 변하는지 보겠습니다.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

그러면 Python 은 다음을 출력합니다.

```
0.12222222222222222
```

감이 오나요? 문자열이 더 비슷할수록 점수가 더 높습니다.

이제 조금 더 큰 예제를 만들어 라이브러리 성능에 대한 감을 잡아 봅시다. Smith-Waterman 과 Needleman-Wunsch 사이를 전환하는 것을 보고 언제 무엇을 써야 하는지 궁금할 수 있습니다. 전체 입력을 전체 입력과 비교하려면 Needleman-Wunsch 를 쓰세요. 더 큰 입력 안에서 가장 잘 맞는 영역을 찾으려면 Smith-Waterman 을 쓰세요.

좋습니다. 데모 코드로 넘어갑니다. 이 부분에는 `requests`가 필요합니다.

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

이걸 어떻게 쓸 수 있을까요? 임의의 성경 구절이 하나 있고, 그것이 어느 장 어느 절인지 알고 싶다고 해 봅시다. `grep`이라고요? 아, 안 됩니다. 우리가 실수를 했습니다. 가진 구절은 Catholic Public Domain 같은 다른 번역본에서 온 것이고, 컴퓨터에는 King James Bible 이 있습니다. `grep`의 정확한 문자열 일치는 여기서 통하지 않습니다. 장과 절을 어떻게 찾을까요? 당연히 `stride-align`으로 “가장 가까운” 또는 “가장 비슷한” 문자열을 찾습니다.

데모의 첫 부분은 다운로드와 캐시를 처리합니다. [Open Bible](https://openbible.com)의 좋은 분들이 이 텍스트를 HTTP 로 접근 가능한 곳에 두었지만, 그들의 IT 예산을 존중해야 하므로 다운로드한 것은 캐시합니다. 그냥 좋은 시민 의식입니다.

다음 부분에서는 모든 줄을 리스트로 읽어 들입니다. 줄바꿈을 없애고 모두 소문자로 만듭니다. Shift 키를 누르고 있었는지 같은 일로 까다롭게 굴고 싶지 않기 때문입니다.

마지막으로 `while True:` 루프가 텍스트 한 줄을 받습니다. 아마 장과 절을 찾고 싶은 Catholic 판 성경 구절일 것입니다. 그리고 Needleman-Wunsch 의 배치 형태를 사용해 King James Bible 의 모든 줄과 맞춥니다. 결과는 점수 배열입니다. `argmax()`로 최고 점수의 줄을 찾고, 그 인덱스에 해당하는 줄을 출력합니다. 해 봅시다.

여기서는 Catholic Bible 의 Jeremiah 4:28 을 쓰겠습니다. King James Bible 의 같은 절과 실제로 꽤 다릅니다. 무슨 일이 일어나는지 보죠.

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

…찾았습니다! 그것도 꽤 빠르게요.

## 또 다른 데모: 맞춤법 검사

이것은 장난감 수준의 맞춤법 검사기이지, 운영 환경용 도구가 아닙니다. 구두점, 대소문자, 단어 빈도, 고유명사, 문맥을 무시합니다. 목적은 익숙한 작업에서 같은 “하나의 질의를 많은 후보와 비교”하는 패턴을 보여 주는 것입니다.

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

이 스크립트가 먼저 하는 일은 운영체제의 올바른 철자 단어 목록을 찾는 것입니다. 위치는 배포판마다 다를 수 있습니다. 찾고 나면 읽어 들이고, 줄바꿈을 제거한 뒤 맞춤법 검사를 시작합니다.

맞춤법 검사는 앞에서 한 매칭과 매우 비슷합니다. 각 후보 단어에 대해 올바른 철자 단어 목록의 모든 단어와 비교하고, `argmax()`를 사용해 가장 높은 점수의 후보를 찾은 뒤, 원래 단어를 그 후보로 바꿉니다. 이미 올바르게 철자된 단어는 검색하지 않는 식의 최적화로 더 빠르게 만들 수 있지만, 이것은 데모이고 그 최적화는 독자 연습문제로 남겨 둡니다.

어떻게 동작하는지 봅시다!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## 세부 사항

현재 골격은 다음을 제공합니다.

- 점수만 계산하는 Needleman-Wunsch 정렬
- 역추적을 포함한 Needleman-Wunsch 정렬
- 점수만 계산하는 Smith-Waterman 정렬
- 역추적을 포함한 Smith-Waterman 정렬
- `massive-speedup`에서 쓰는 특수화 패턴과 맞는 백엔드 배치
- CPU/백엔드 감지 및 Python 쪽 백엔드 분배

네이티브 경계는 다음을 받습니다.

- `bytes` 대 `bytes`
- `str` 대 `str`
- 불변이고 해시 가능한 Python 객체의 시퀀스
- 혼합된 시퀀스/객체 입력. `str` 또는 `bytes` 쪽은 시퀀스로 취급됩니다

직접적인 `bytes` 대 `str` 쌍은 `TypeError`를 발생시킵니다.

현재 구현은 전처리가 있는 일반 동적 계획법 커널입니다. 전처리는 Python 입력을 8, 16, 32 또는 64비트 토큰 스트림으로 직렬화합니다. SIMD 특수화 백엔드는 Python API 를 바꾸지 않고 나중에 백엔드 번역 단위를 교체할 수 있습니다.

점수만 반환하는 함수는 숫자 점수를 반환합니다. 정규화 변형은 `0`과 `1` 사이의 점수를 반환합니다. 경로 함수는 점수, 정렬된 값, 연산, 그리고 가능한 경우 CIGAR 스타일 요약을 포함한 정렬 결과 객체를 반환합니다.

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

하나의 질의를 많은 대상에 비교하는 점수 작업에는 `Scores(...).compare([...])` 또는 `*_scores()` 함수를 사용하세요. 이 경로는 질의/프로파일을 한 번만 준비하므로 반복적인 영어/중국어 텍스트 비교에 권장되는 고성능 API 입니다.

역추적 출력은 짝을 이룬 빠른 경로 타입을 보존합니다.

- `str` 입력은 정렬된 `str`을 반환합니다
- `bytes` 입력은 정렬된 `bytes`를 반환합니다
- 시퀀스/객체 입력은 `None` 간격이 있는 정렬된 `tuple` 값을 반환합니다

`width=8`, `16`, `32`, 또는 `64`를 전달하면 자동 선택 대신 내부 토큰/채점 폭을 강제로 지정할 수 있습니다.

일부 함수는 CIGAR 문자열을 노출합니다. CIGAR 는 “Concise Idiosyncratic Gapped Alignment Report”의 약자입니다. SAM/BAM 도구에서 쓰는 간결한 정렬 연산 표기입니다. 완전한 공식 버전이 필요하면 [SAM 명세](https://samtools.github.io/hts-specs/SAMv1.pdf)를 보세요.

## 최적화와 벤치마크

`stride-align`의 성능 이야기는 지금까지도, 앞으로도 신중하게 다룹니다. 이 라이브러리는 x86, Arm, LoongArch 를 포함한 여러 일반 대상에 대한 SIMD 최적화를 포함합니다.

**LoongArch / Loongson.** Loongson 최적화 이야기는 특히 분명합니다. 확인된 벤치마크 사례 중 영어 텍스트, 16비트 점수 폭, 점수 전용 Smith-Waterman 계산에서는 LASX 백엔드가 일반 백엔드보다 16배 빠르고 Parasail보다 **22.4배** 빠릅니다.

Loongson 서버를 사용하는 연구자이고 이 속도 향상의 혜택을 보고 있다면, 인용, 버그 보고서, 벤치마크 사례, 그리고 작고 저렴한 중국 기념품을 환영합니다. 차, 서예 책갈피, 종이 공예 장식, 중국 매듭 장식, 판다 열쇠고리, 작은 용 모양 책상 장식 모두 좋습니다. 비싼 물건이나 세관 서류가 필요한 것은 보내지 마세요.

[전체 벤치마크](BENCHMARKS.md)를 보세요.

## 네이티브 마이크로벤치

Python 프레임이나 벤치마크 오케스트레이션 없이 성능 프로파일링을 하려면 네이티브 x86 마이크로벤치 빌드를 구성하세요.

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON`은 nanobind 모듈을 제거하지 않은 상태로 유지하고 디버그 심볼과 프레임 포인터를 추가하면서 `-O3`를 보존합니다.

체크인된 네이티브 마이크로벤치 기준선은 `benchmarks/x86_microbench_baseline.json`에 있습니다. 이것은 느슨한 임계값을 가진 로컬 안전장치로 취급하고, 여러 머신에 걸친 SLA 로 취급하지 마세요.


## 인용

연구에서 제 소프트웨어를 사용한다면 저를 인용해 주세요.

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
