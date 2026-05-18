# stride-align

**Języki:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · **[Polski](README.pl.md)**

`stride-align` to biblioteka, która mówi, jak bardzo „podobne” są dwa łańcuchy znaków. Robi to przez implementację algorytmów Smith-Waterman i Needleman-Wunsch. Zamiast wykładu będziemy uczyć się przez działanie. Wejdźmy od razu w to, jak to działa.

## Instalacja

```bash
pip install stride-align
```

Na systemach Loongson zainstaluj NumPy z własnej dystrybucji Linuksa przed instalacją `stride-align`:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Najpierw małe zastrzeżenie: nie używam tutaj tekstów religijnych, żeby promować jakąkolwiek agendę. Do tej demonstracji potrzebuję kilku dość dużych dokumentów w domenie publicznej, które mają to samo znaczenie, ale są sformułowane inaczej. Biblia po prostu dziwnie dobrze pasuje do tego wymagania.

Wyobraźmy sobie, że mamy dwa zdania; użyjmy pierwszego zdania z Genesis:

W American Standard Version mamy: "In the beginning God created the heavens and the earth."

W King James Version mamy: "In the beginning God created the heaven and the earth."

Na oko widać różnicę: heavens kontra heaven. Ale jak tę różnicę zmierzyć liczbowo? Użyjemy tego małego kawałka kodu:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Po uruchomieniu wypisuje:

```python
0.9907407407407407
```

Znormalizowane wyniki są między `0` a `1`. Wynik `1` oznacza, że wejścia są dokładnym dopasowaniem w domyślnym modelu punktacji. Wyniki bliskie `0` oznaczają, że wejścia mają niewiele wspólnego, choć Smith-Waterman może nadal znaleźć małe lokalne dopasowania wewnątrz poza tym niepowiązanych łańcuchów.

Zmieńmy teraz tekst i zobaczmy, co stanie się z wynikiem.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

a Python wypisuje:

```
0.12222222222222222
```

Zaczyna być jasne? Im bardziej podobne łańcuchy, tym wyższy wynik.

Zbudujmy większy przykład, który da nam wyczucie wydajności biblioteki. Pewnie zauważysz, że przełączamy się między Smith-Waterman i Needleman-Wunsch, i możesz zastanawiać się, kiedy używać którego. Używaj Needleman-Wunsch, gdy chcesz porównać całe wejście z całym wejściem. Używaj Smith-Waterman, gdy chcesz znaleźć najlepiej pasujący region wewnątrz większych wejść.

Dobrze, przejdźmy do kodu demonstracyjnego. Do tej części potrzebujesz `requests`:

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

Jak możemy tego użyć? Załóżmy, że mamy losowy werset biblijny i chcemy wiedzieć, z którego rozdziału i wersetu pochodzi. `grep`, mówisz? O nie, popełniliśmy błąd. Werset, który mamy, pochodzi z innego tłumaczenia, powiedzmy Catholic Public Domain, a na komputerze mamy King James Bible. Dokładne dopasowanie łańcuchów w `grep` tutaj nie zadziała. Jak znaleźć rozdział i werset? Oczywiście szukamy „najbliższego” albo „najbardziej podobnego” łańcucha za pomocą `stride-align`.

W naszej demonstracji pierwsza część zajmuje się pobieraniem i buforowaniem. Dobrzy ludzie z [Open Bible](https://openbible.com) umieścili ten tekst tak, że jest dostępny przez HTTP, ale chcemy szanować ich budżet IT, więc buforujemy to, co pobieramy. To po prostu dobre obywatelstwo sieciowe.

W następnej części ładujemy wszystkie wiersze do listy. Usuwamy znaki nowej linii i zamieniamy wszystko na małe litery, bo nie chcemy się drobiazgowo spierać o to, czy ktoś trzymał Shift.

Na koniec pętla `while True:` zbiera jedną linię tekstu, przypuszczalnie werset z Catholic Bible, dla którego chcemy znaleźć rozdział i werset, i dopasowuje go do wszystkich linii King James Bible przy użyciu wsadowej postaci Needleman-Wunsch. Zwraca tablicę wyników. Używamy `argmax()`, aby znaleźć najlepiej punktowaną linię, a następnie wypisujemy linię powiązaną z tym indeksem. Spróbujmy.

Użyję Jeremiah 4:28 z Catholic Bible; jest naprawdę dość różny od tego samego wersetu w King James Bible. Zobaczmy, co się stanie…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

…i znaleźliśmy go! I całkiem szybko.

## Kolejna demonstracja: sprawdzanie pisowni

To zabawkowy sprawdzacz pisowni, nie produkcyjny. Ignoruje interpunkcję, wielkość liter, częstotliwość słów, nazwy własne i kontekst. Chodzi o pokazanie tego samego wzorca „jedno zapytanie przeciw wielu kandydatom” na znajomym zadaniu.

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

Pierwszą rzeczą, którą robi ten skrypt, jest próba znalezienia listy poprawnie napisanych słów w systemie operacyjnym. Jej położenie może się różnić między dystrybucjami. Gdy ją znajdziemy, ładujemy ją, usuwamy znaki nowej linii i zaczynamy sprawdzanie pisowni.

Sprawdzanie pisowni wygląda bardzo podobnie do dopasowania, które robiliśmy wcześniej. Dla każdego słowa kandydującego dopasowujemy je do wszystkich słów z naszej listy poprawnie napisanych słów, używamy `argmax()`, aby znaleźć kandydata z najwyższym wynikiem, i zastępujemy słowo tym kandydatem. Moglibyśmy to przyspieszyć optymalizacjami, na przykład nie szukać dopasowania dla słów już poprawnych, ale to jest demonstracja i ta optymalizacja zostaje jako ćwiczenie dla czytelnika.

Zobaczmy, jak działa!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Szczegóły

Obecny szkielet dostarcza:

- wyrównanie Needleman-Wunsch tylko z wynikiem
- wyrównanie Needleman-Wunsch z odtwarzaniem ścieżki
- wyrównanie Smith-Waterman tylko z wynikiem
- wyrównanie Smith-Waterman z odtwarzaniem ścieżki
- układ backendów pasujący do wzorca specjalizacji użytego w `massive-speedup`
- wykrywanie CPU/backendu i dyspozycję backendu po stronie Python

Granica natywna akceptuje:

- `bytes` przeciw `bytes`
- `str` przeciw `str`
- sekwencje niezmiennych, haszowalnych obiektów Python
- mieszane wejścia sekwencja/obiekt, gdzie strona `str` lub `bytes` jest traktowana jako sekwencja

Bezpośrednie pary `bytes` przeciw `str` zgłaszają `TypeError`.

Obecne implementacje to ogólne jądra programowania dynamicznego z przetwarzaniem wstępnym, które serializuje wejścia Python do strumieni tokenów 8-, 16-, 32- lub 64-bitowych. Backendy wyspecjalizowane pod SIMD mogą później zastąpić jednostki translacji backendu bez zmiany Python API.

Funkcje zwracające tylko wynik zwracają wyniki liczbowe. Warianty znormalizowane zwracają wyniki między `0` a `1`. Funkcje ścieżki zwracają obiekty wyniku wyrównania zawierające wynik, wyrównane wartości, operacje oraz, gdy dostępne, podsumowania w stylu CIGAR.

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

Użyj `Scores(...).compare([...])` albo funkcji `*_scores()` dla obciążeń typu jedno zapytanie przeciw wielu celom. Ta ścieżka przygotowuje zapytanie/profil raz i jest preferowanym wydajnościowym API dla powtarzanych porównań tekstów angielskich/chińskich.

Wyjścia odtwarzania ścieżki zachowują sparowany typ szybkiej ścieżki:

- `str` wejścia zwracają wyrównane `str`
- `bytes` wejścia zwracają wyrównane `bytes`
- wejścia sekwencja/obiekt zwracają wyrównane wartości `tuple` z lukami `None`

Przekaż `width=8`, `16`, `32` lub `64`, aby wymusić wewnętrzną szerokość tokenów/punktacji zamiast automatycznego wyboru.

Niektóre funkcje udostępniają łańcuchy CIGAR, skrót od “Concise Idiosyncratic Gapped Alignment Report”. CIGAR to zwarta notacja operacji wyrównania używana przez narzędzia SAM/BAM. Jeśli chcesz pełną formalną wersję, zobacz [specyfikację SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Optymalizacje i benchmarki

Historii wydajności `stride-align` poświęcano i nadal poświęca się dużo uwagi. Biblioteka zawiera optymalizacje SIMD dla wielu typowych celów, w tym x86, Arm i LoongArch.

**LoongArch / Loongson.** Historia optymalizacji Loongson jest szczególnie wymowna: dla obliczeń tylko wyniku na sprawdzonym benchmarku backend LASX jest 16 razy szybszy od backendu ogólnego i **22,3 razy** szybszy od Parasail.

Jeśli jesteś badaczem używającym serwerów Loongson i korzystasz z tego przyspieszenia, mile widziane są cytowania, zgłoszenia błędów, przypadki benchmarkowe i małe niedrogie chińskie pamiątki. Herbata, zakładki kaligraficzne, papierowe wycinanki, ozdoby z chińskim węzłem, breloczki z pandą i małe smocze przedmioty na biurko są mile widziane. Proszę nie wysyłać niczego drogiego ani niczego, co wymaga dokumentów celnych.

Zobacz [pełne benchmarki](BENCHMARKS.md).

## Natywny mikrobenchmark

Do profilowania wydajności bez ramek Python i bez orkiestracji benchmarków skonfiguruj natywną kompilację mikrobenchmarku x86:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` zachowuje moduły nanobind bez usuwania symboli i dodaje symbole debugowania oraz wskaźniki ramek, jednocześnie zachowując `-O3`.

Zapisana w repozytorium linia bazowa natywnego mikrobenchmarku znajduje się w `benchmarks/x86_microbench_baseline.json`. Traktuj ją jako lokalną barierę z luźnym progiem, a nie jako SLA między maszynami.


## Cytowania

Jeśli używasz mojego oprogramowania w swoich badaniach, proszę zacytuj mnie.

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
