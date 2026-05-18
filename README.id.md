# stride-align

**Bahasa:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · **[Bahasa Indonesia](README.id.md)** · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` adalah [pustaka yang sangat cepat](BENCHMARK.md) untuk memberi tahu seberapa “mirip” dua string. Pustaka ini melakukannya dengan mengimplementasikan algoritma Smith-Waterman dan Needleman-Wunsch. Alih-alih memberi kuliah, kita akan belajar dengan mencoba. Mari langsung masuk ke cara kerjanya.

## Instalasi

```bash
pip install stride-align
```

Pada sistem Loongson, instal NumPy dari distribusi Linux Anda sebelum memasang `stride-align`:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Pertama, sedikit penjelasan: saya tidak memakai teks keagamaan di sini untuk mendorong agenda apa pun. Untuk demo ini saya membutuhkan beberapa dokumen domain publik yang cukup besar, memiliki makna yang sama, tetapi dirumuskan berbeda. Alkitab kebetulan cocok dengan kebutuhan demo itu secara hampir menggelikan.

Bayangkan kita punya dua kalimat; untuk ini mari pakai kalimat pertama dalam Genesis:

Dalam American Standard Version kita punya: "In the beginning God created the heavens and the earth."

Dalam King James Version kita punya: "In the beginning God created the heaven and the earth."

Dengan mata kita melihat ada perbedaan: heavens dibanding heaven. Tetapi bagaimana kita mengukur perbedaan itu? Kita memakai sedikit kode ini:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Saat dijalankan, ia mencetak:

```python
0.9907407407407407
```

Skor ternormalisasi berada antara `0` dan `1`. Skor `1` berarti masukan cocok persis di bawah model penilaian bawaan. Skor mendekati `0` berarti masukan punya sedikit kesamaan, meskipun Smith-Waterman masih bisa menemukan kecocokan lokal kecil di dalam string yang selebihnya tidak berkaitan.

Sekarang mari ubah teks dan lihat apa yang terjadi pada skornya.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

dan Python mencetak:

```
0.12222222222222222
```

Mulai paham idenya? Semakin mirip string, semakin tinggi skornya.

Mari buat contoh yang lebih besar, sesuatu yang memberi kita rasa atas performa pustaka ini. Anda mungkin memperhatikan bahwa kita berganti antara Smith-Waterman dan Needleman-Wunsch, dan mungkin bertanya kapan harus memakai yang mana. Gunakan Needleman-Wunsch saat ingin membandingkan seluruh masukan dengan seluruh masukan. Gunakan Smith-Waterman saat ingin menemukan wilayah yang paling cocok di dalam masukan yang lebih besar.

Oke, mari lanjut ke kode demo. Anda membutuhkan `requests` untuk bagian demo ini:

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

Bagaimana kita memakai ini? Misalkan kita punya ayat Alkitab acak dan ingin tahu dari pasal dan ayat mana asalnya. `grep`, kata Anda? Aduh, tidak, kita membuat kesalahan. Ayat yang kita punya berasal dari terjemahan lain, misalnya Catholic Public Domain, sedangkan di komputer kita ada King James Bible. Pencocokan string persis milik `grep` tidak akan bekerja di sini. Bagaimana menemukan pasal dan ayatnya? Tentu saja kita mencari string yang “paling dekat” atau “paling mirip” memakai `stride-align`.

Dalam demo kita, bagian pertama menangani pengunduhan dan cache. Orang-orang baik di [Open Bible](https://openbible.com) menaruh teks ini di tempat yang dapat dijangkau lewat HTTP, tetapi kita ingin menghormati anggaran TI mereka, jadi kita menyimpan hasil unduhan ke cache. Itu hanya kewargaan internet yang baik.

Pada bagian berikutnya kita memuat semua baris ke dalam daftar. Kita menghapus baris baru dan membuat semuanya huruf kecil karena tidak ingin rewel soal apakah tombol Shift sedang ditekan.

Terakhir, loop `while True:` mengambil satu baris teks, mungkin ayat dari versi Catholic Bible yang ingin kita cari pasal dan ayatnya, lalu mencocokkannya dengan semua baris dalam King James Bible memakai bentuk batch dari Needleman-Wunsch. Ia mengembalikan array skor. Kita memakai `argmax()` untuk menemukan baris dengan skor tertinggi, lalu mencetak baris yang terkait dengan indeks itu. Mari coba.

Saya akan memakai Jeremiah 4:28 dari Catholic Bible; sebenarnya cukup berbeda dari ayat yang sama dalam King James Bible. Mari lihat apa yang terjadi…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

… dan kita menemukannya! Cukup cepat pula.

## Demo lain: pemeriksaan ejaan

Ini pemeriksa ejaan mainan, bukan alat produksi. Ia mengabaikan tanda baca, kapitalisasi, frekuensi kata, nama diri, dan konteks. Tujuannya adalah menunjukkan pola satu-kueri-melawan-banyak-kandidat yang sama pada tugas yang akrab.

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

Hal pertama yang dilakukan skrip ini adalah mencoba menemukan daftar kata yang dieja dengan benar milik sistem operasi. Lokasinya bisa berbeda dari satu distribusi ke distribusi lain. Setelah ditemukan, kita memuatnya, menghapus baris baru, dan mulai pemeriksaan ejaan.

Pemeriksaan ejaan sangat mirip dengan pencocokan yang kita lakukan sebelumnya. Untuk setiap kata kandidat, kita mencocokkannya dengan semua kata dalam daftar kata yang benar, memakai `argmax()` untuk menemukan kandidat berskor tertinggi, lalu mengganti kata itu dengan kandidat tersebut. Kita bisa mempercepatnya dengan optimasi, misalnya tidak mencari kecocokan untuk kata yang sudah benar, tetapi ini demo dan optimasi itu ditinggalkan sebagai latihan untuk pembaca.

Mari lihat cara kerjanya!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Detail

Kerangka saat ini menyediakan:

- penyelarasan Needleman-Wunsch hanya skor
- penyelarasan Needleman-Wunsch dengan penelusuran balik
- penyelarasan Smith-Waterman hanya skor
- penyelarasan Smith-Waterman dengan penelusuran balik
- tata letak backend yang cocok dengan pola spesialisasi yang digunakan di `massive-speedup`
- deteksi CPU/backend dan pengiriman backend sisi Python

Batas native menerima:

- `bytes` melawan `bytes`
- `str` melawan `str`
- urutan objek Python yang tidak dapat diubah dan dapat di-hash
- masukan campuran urutan/objek, dengan sisi `str` atau `bytes` diperlakukan sebagai urutan

Pasangan langsung `bytes` melawan `str` memunculkan `TypeError`.

Implementasi saat ini adalah kernel pemrograman dinamis generik dengan prapemrosesan yang menserialkan masukan Python menjadi aliran token 8, 16, 32, atau 64 bit. Backend terspesialisasi SIMD dapat mengganti unit terjemahan backend nanti tanpa mengubah API Python.

Fungsi hanya-skor mengembalikan skor numerik. Varian ternormalisasi mengembalikan skor antara `0` dan `1`. Fungsi jalur mengembalikan objek hasil penyelarasan yang berisi skor, nilai terselaras, operasi, dan ringkasan bergaya CIGAR jika tersedia.

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

Gunakan `Scores(...).compare([...])` atau fungsi `*_scores()` untuk beban kerja satu kueri melawan banyak target. Jalur itu menyiapkan kueri/profil satu kali dan merupakan API performa yang disukai untuk perbandingan teks Inggris/Tionghoa berulang.

Keluaran penelusuran balik mempertahankan tipe jalur cepat yang berpasangan:

- `str` masukan mengembalikan `str` yang terselaras
- `bytes` masukan mengembalikan `bytes` yang terselaras
- masukan urutan/objek mengembalikan nilai `tuple` terselaras dengan celah `None`

Berikan `width=8`, `16`, `32`, atau `64` untuk memaksa lebar token/penilaian internal alih-alih memakai pemilihan otomatis.

Beberapa fungsi mengekspos string CIGAR, singkatan dari “Concise Idiosyncratic Gapped Alignment Report”. CIGAR adalah notasi ringkas operasi penyelarasan yang dipakai alat SAM/BAM. Jika ingin versi formal lengkap, lihat [spesifikasi SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Optimasi dan benchmark

Performa `stride-align` telah, dan terus, mendapat perhatian serius. Pustaka ini menyertakan optimasi SIMD untuk berbagai target umum, termasuk x86, Arm, dan LoongArch.

**LoongArch / Loongson.** Kisah optimasi Loongson sangat menjelaskan: pada kasus benchmark yang diperiksa untuk teks bahasa Inggris, lebar skor 16-bit, dan Smith-Waterman hanya-skor, backend LASX 16 kali lebih cepat daripada backend generik dan **22,4 kali** lebih cepat daripada Parasail.

Jika Anda peneliti yang memakai server Loongson dan mendapat manfaat dari percepatan ini, sitasi, laporan bug, kasus benchmark, dan suvenir kecil Tiongkok yang murah sangat dihargai. Teh, pembatas buku kaligrafi, ornamen guntingan kertas, jimat simpul Tiongkok, gantungan kunci panda, dan benda meja kecil berbentuk naga semuanya disambut. Tolong jangan kirim apa pun yang mahal atau membutuhkan dokumen bea cukai.

Lihat [benchmark lengkap](BENCHMARKS.md).

## Microbench native

Untuk profiling performa tanpa frame Python atau orkestrasi benchmark, konfigurasikan build microbench x86 native:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` menjaga modul nanobind tidak di-strip dan menambahkan simbol debug serta frame pointer sambil mempertahankan `-O3`.

Baseline microbench native yang masuk repositori berada di `benchmarks/x86_microbench_baseline.json`. Perlakukan sebagai pagar lokal dengan ambang longgar, bukan SLA lintas-mesin.


## Sitasi

Jika Anda memakai perangkat lunak saya dalam riset Anda, mohon sitasi saya.

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
