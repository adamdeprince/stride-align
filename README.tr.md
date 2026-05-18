# stride-align

**Diller:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · **[Türkçe](README.tr.md)** · [Polski](README.pl.md)

`stride-align` iki dizgenin ne kadar “benzer” olduğunu söyleyen [alev gibi hızlı bir kütüphanedir](BENCHMARK.md). Bunu Smith-Waterman ve Needleman-Wunsch algoritmalarını uygulayarak yapar. Ders anlatmak yerine yaparak öğreneceğiz. Hemen nasıl çalıştığına bakalım.

## Kurulum

```bash
pip install stride-align
```

Loongson sistemlerinde `stride-align` kurmadan önce NumPy’yi Linux dağıtımınızdan kurun:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Önce küçük bir not: Burada dini metinleri bir gündem dayatmak için kullanmıyorum. Bu demo için aynı anlama gelen ama farklı ifade edilmiş, kamu malı, birkaç büyükçe belgeye ihtiyacım var. İncil bu demo gereksinimine acayip derecede iyi uyuyor.

İki cümlemiz olduğunu düşünün; bunun için Genesis’in ilk cümlesini kullanalım:

American Standard Version’da şunu görüyoruz: "In the beginning God created the heavens and the earth."

King James Version’da şunu görüyoruz: "In the beginning God created the heaven and the earth."

Gözümüzle farkı görüyoruz: heavens ile heaven. Peki bu farkı nasıl sayısallaştırırız? Şu küçük kod parçasını kullanırız:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Bunu çalıştırınca şunu yazdırır:

```python
0.9907407407407407
```

Normalize puanlar `0` ile `1` arasındadır. `1` puanı, varsayılan puanlama modelinde girdilerin tam eşleştiği anlamına gelir. `0`’a yakın puanlar girdilerin çok az ortak noktası olduğunu gösterir; yine de Smith-Waterman, başka türlü alakasız dizgelerin içinde küçük yerel eşleşmeler bulabilir.

Şimdi metni değiştirip puanın ne olduğuna bakalım.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

ve Python şunu yazdırır:

```
0.12222222222222222
```

Fikir oturmaya başladı mı? Dizgeler ne kadar benzerse puan o kadar yüksek olur.

Şimdi daha büyük bir örnek kuralım; kütüphanenin performansı hakkında bir his versin. Smith-Waterman ve Needleman-Wunsch arasında geçiş yaptığımızı fark edeceksiniz ve hangisini ne zaman kullanacağınızı merak edebilirsiniz. Tüm girdiyi tüm girdiyle karşılaştırmak istediğinizde Needleman-Wunsch kullanın. Daha büyük girdilerin içinde en iyi eşleşen bölgeyi bulmak istediğinizde Smith-Waterman kullanın.

Tamam, demo koduna geçelim. Bu bölüm için `requests` gerekir:

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

Bunu nasıl kullanırız? Rastgele bir İncil ayetimiz olduğunu ve bunun hangi bölüm ve ayetten geldiğini bilmek istediğimizi varsayalım. `grep` mi diyorsunuz? Ah, hayır, hata yaptık. Elimizdeki ayet başka bir çeviriden, örneğin Catholic Public Domain’den geliyor; bilgisayardaki metin ise King James Bible. `grep`in tam dizge eşleştirmesi burada işe yaramaz. Bölüm ve ayeti nasıl buluruz? Elbette `stride-align` ile “en yakın” veya “en benzer” dizgeyi ararız.

Demomuzda ilk kısım indirme ve önbellekleme ile ilgilenir. [Open Bible](https://openbible.com) sitesindeki iyi insanlar bu metni HTTP ile erişilebilir bir yere koymuşlar, ama onların BT bütçesine saygılı olmak istiyoruz; bu yüzden indirdiklerimizi önbelleğe alıyoruz. Bu basitçe iyi vatandaşlık.

Sonraki bölümde tüm satırları bir listeye yüklüyoruz. Satır sonlarını kaldırıyor ve her şeyi küçük harfe çeviriyoruz, çünkü Shift tuşuna basılıp basılmadığı konusunda huysuzlanmak istemiyoruz.

Son olarak `while True:` döngüsü bir metin satırı toplar; muhtemelen bölüm ve ayetini aradığımız Catholic Bible ayeti. Bunu Needleman-Wunsch’ın toplu biçimini kullanarak King James Bible’daki tüm satırlarla eşleştirir. Bir puan dizisi döndürür. En iyi puanlı satırı bulmak için `argmax()` kullanır, sonra o indeksle ilişkili satırı yazdırırız. Deneyelim.

Catholic Bible’dan Jeremiah 4:28 kullanacağım; King James Bible’daki aynı ayetten gerçekten oldukça farklı. Bakalım ne olacak…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

…ve bulduk! Üstelik oldukça hızlı.

## Başka bir demo: yazım denetimi

Bu oyuncak bir yazım denetleyicisidir, üretim aracı değildir. Noktalama, büyük/küçük harf, kelime sıklığı, özel adlar ve bağlamı yok sayar. Amaç, aynı “bir sorguya karşı çok aday” örüntüsünü tanıdık bir görevde göstermektir.

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

Bu betiğin yaptığı ilk şey, işletim sistemimizin doğru yazılmış kelime listesini bulmaya çalışmaktır. Konumu dağıtımdan dağıtıma değişebilir. Bulunca yükler, satır sonlarını kırpar ve yazım denetimi işlemine başlarız.

Yazım denetimi, daha önce yaptığımız eşleştirmeye çok benzer. Her aday kelime için onu doğru yazılmış kelime listemizdeki tüm kelimelerle eşleştirir, en yüksek puanlı adayı bulmak için `argmax()` kullanır ve kelimeyi o adayla değiştiririz. Doğru yazılmış kelimeler için eşleşme aramamak gibi optimizasyonlarla hızlandırabiliriz, ama bu bir demo ve o optimizasyon okuyucuya alıştırma olarak bırakılmıştır.

Nasıl çalıştığını görelim!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Ayrıntılar

Mevcut iskelet şunları sağlar:

- yalnızca puan veren Needleman-Wunsch hizalaması
- geri izlemeli Needleman-Wunsch hizalaması
- yalnızca puan veren Smith-Waterman hizalaması
- geri izlemeli Smith-Waterman hizalaması
- `massive-speedup` içinde kullanılan özelleştirme örüntüsüyle eşleşen bir backend yerleşimi
- CPU/backend algılama ve Python tarafında backend dağıtımı

Yerel sınır şunları kabul eder:

- `bytes` ile `bytes`
- `str` ile `str`
- değişmez ve hashlenebilir Python nesnelerinin dizileri
- `str` veya `bytes` tarafının dizi olarak ele alındığı karma dizi/nesne girdileri

Doğrudan `bytes` ile `str` çiftleri `TypeError` yükseltir.

Mevcut uygulamalar, Python girdilerini 8, 16, 32 veya 64 bitlik belirteç akışlarına serileştiren ön işlemeli genel dinamik programlama çekirdekleridir. SIMD’ye özelleştirilmiş backend’ler daha sonra Python API’sini değiştirmeden backend çeviri birimlerini değiştirebilir.

Yalnızca puan veren fonksiyonlar sayısal puan döndürür. Normalize varyantlar `0` ile `1` arasında puan döndürür. Yol fonksiyonları puanı, hizalanmış değerleri, işlemleri ve varsa CIGAR tarzı özetleri içeren hizalama sonuç nesneleri döndürür.

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

Bir sorguyu çok hedefe karşı puanlama iş yükleri için `Scores(...).compare([...])` veya `*_scores()` fonksiyonlarını kullanın. Bu yol sorgu/profili bir kez hazırlar ve tekrarlanan İngilizce/Çince metin karşılaştırmaları için tercih edilen performans API’sidir.

Geri izleme çıktıları eşleşmiş hızlı yol türünü korur:

- `str` girdileri hizalanmış `str` döndürür
- `bytes` girdileri hizalanmış `bytes` döndürür
- dizi/nesne girdileri `None` boşlukları içeren hizalanmış `tuple` değerleri döndürür

Otomatik seçim yerine iç belirteç/puanlama genişliğini zorlamak için `width=8`, `16`, `32` veya `64` geçirin.

Bazı fonksiyonlar CIGAR dizgeleri sunar; CIGAR, “Concise Idiosyncratic Gapped Alignment Report” kısaltmasıdır. CIGAR, SAM/BAM araçlarının kullandığı kompakt hizalama-işlem gösterimidir. Tam resmi sürüm için [SAM belirtimine](https://samtools.github.io/hts-specs/SAMv1.pdf) bakın.

## Optimizasyonlar ve benchmarklar

`stride-align`ın performans hikâyesine dikkatle bakıldı ve bakılmaya devam ediyor. Kütüphane x86, Arm ve LoongArch dahil çeşitli yaygın hedefler için SIMD optimizasyonu içerir.

**LoongArch / Loongson.** Loongson optimizasyon hikâyesi özellikle açıklayıcıdır: kontrol edilen benchmark vakasında, İngilizce metin, 16 bit puan genişliği ve yalnızca puan hesaplayan Smith-Waterman için LASX backend’i genel backend’den 16 kat, Parasail’den **22,4 kat** daha hızlıdır.

Loongson sunucuları kullanan ve bu hızlanmadan yararlanan bir araştırmacıysanız, atıflar, hata raporları, benchmark örnekleri ve küçük ucuz Çin hediyelikleri makbuldür. Çay, kaligrafi ayraçları, kâğıt kesme süsleri, Çin düğümü süsleri, panda anahtarlıkları ve küçük ejderha masa objeleri hoş karşılanır. Lütfen pahalı bir şey veya gümrük evrakı gerektiren bir şey göndermeyin.

[Tam benchmarklara](BENCHMARKS.md) bakın.

## Yerel mikrobenchmark

Python frame’leri veya benchmark orkestrasyonu olmadan performans profilleme için yerel x86 mikrobenchmark derlemesi yapılandırın:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON`, nanobind modüllerini sembolleri sıyrılmamış halde tutar ve `-O3`ü korurken hata ayıklama sembolleri ile frame pointer’ları ekler.

Depoya konmuş yerel mikrobenchmark temel çizgisi `benchmarks/x86_microbench_baseline.json` içindedir. Bunu gevşek eşikli yerel bir korkuluk olarak görün, makineler arası SLA olarak değil.


## Atıflar

Yazılımımı araştırmanızda kullanırsanız lütfen bana atıf verin.

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
