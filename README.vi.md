# stride-align

**Ngôn ngữ:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · **[Tiếng Việt](README.vi.md)** · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` là một [thư viện nhanh rực lửa](BENCHMARK.md) cho bạn biết hai chuỗi “giống nhau” đến mức nào. Nó làm việc đó bằng cách triển khai các thuật toán Smith-Waterman và Needleman-Wunsch. Thay vì giảng bài, ta sẽ học bằng cách làm. Đi thẳng vào cách nó hoạt động.

## Cài đặt

```bash
pip install stride-align
```

Trên hệ thống Loongson, hãy cài NumPy từ bản phân phối Linux của bạn trước khi cài `stride-align`:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Trước hết, nói rõ một chút: tôi không dùng văn bản tôn giáo ở đây để thúc đẩy bất kỳ chương trình nghị sự nào. Với bản demo này tôi cần nhiều tài liệu khá lớn, thuộc phạm vi công cộng, có cùng ý nghĩa nhưng diễn đạt khác nhau. Kinh Thánh tình cờ đáp ứng yêu cầu đó một cách kỳ lạ.

Hãy tưởng tượng ta có hai câu; dùng câu đầu tiên trong Genesis cho ví dụ này:

Trong American Standard Version, ta có: "In the beginning God created the heavens and the earth."

Trong King James Version, ta có: "In the beginning God created the heaven and the earth."

Nhìn bằng mắt ta thấy có khác biệt: heavens so với heaven. Nhưng làm sao định lượng khác biệt đó? Dùng đoạn mã nhỏ này:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Khi chạy, nó in ra:

```python
0.9907407407407407
```

Điểm chuẩn hóa nằm giữa `0` và `1`. Điểm `1` nghĩa là các đầu vào khớp chính xác theo mô hình chấm điểm mặc định. Điểm gần `0` nghĩa là các đầu vào có rất ít điểm chung, dù Smith-Waterman vẫn có thể tìm các khớp cục bộ nhỏ bên trong những chuỗi nhìn chung không liên quan.

Bây giờ đổi văn bản và xem điểm thay đổi thế nào.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

và Python in ra:

```
0.12222222222222222
```

Bắt đầu thấy ý tưởng chưa? Chuỗi càng giống nhau thì điểm càng cao.

Hãy xây một ví dụ lớn hơn để có cảm giác về hiệu năng của thư viện. Bạn có thể nhận ra ta chuyển giữa Smith-Waterman và Needleman-Wunsch và thắc mắc khi nào dùng cái nào. Dùng Needleman-Wunsch khi bạn muốn so sánh toàn bộ đầu vào với toàn bộ đầu vào. Dùng Smith-Waterman khi bạn muốn tìm vùng khớp tốt nhất bên trong các đầu vào lớn hơn.

Được rồi, chuyển sang mã demo. Bạn cần `requests` cho phần này:

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

Ta dùng nó thế nào? Giả sử có một câu Kinh Thánh ngẫu nhiên và muốn biết nó đến từ chương và câu nào. Bạn nói `grep`? Ôi không, ta đã mắc lỗi. Câu đó đến từ một bản dịch khác, chẳng hạn Catholic Public Domain, còn trên máy ta có King James Bible. So khớp chuỗi chính xác của `grep` không dùng được ở đây. Vậy tìm chương và câu thế nào? Tất nhiên là tìm chuỗi “gần nhất” hoặc “giống nhất” bằng `stride-align`.

Trong demo, phần đầu xử lý tải xuống và lưu bộ nhớ đệm. Những người tốt ở [Open Bible](https://openbible.com) đặt văn bản này ở nơi truy cập được qua HTTP, nhưng ta muốn tôn trọng ngân sách IT của họ nên lưu đệm những gì đã tải. Đó chỉ là phép lịch sự trên mạng.

Phần tiếp theo nạp tất cả các dòng vào một danh sách. Ta bỏ ký tự xuống dòng và chuyển mọi thứ về chữ thường vì không muốn lăn tăn chuyện có đang giữ phím Shift hay không.

Cuối cùng, vòng lặp `while True:` nhận một dòng văn bản, có lẽ là câu trong bản Catholic của Kinh Thánh mà ta muốn tra chương và câu, rồi so khớp nó với mọi dòng trong King James Bible bằng dạng xử lý hàng loạt của Needleman-Wunsch. Nó trả về một mảng điểm. Ta dùng `argmax()` để tìm dòng có điểm cao nhất rồi in dòng tương ứng với chỉ số đó. Thử nhé.

Tôi sẽ dùng Jeremiah 4:28 từ Catholic Bible; thực ra nó khá khác so với cùng câu trong King James Bible. Xem chuyện gì xảy ra…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

… và tìm thấy rồi! Cũng khá nhanh.

## Demo khác: kiểm tra chính tả

Đây là bộ kiểm tra chính tả đồ chơi, không phải công cụ sản xuất. Nó bỏ qua dấu câu, chữ hoa/chữ thường, tần suất từ, danh từ riêng và ngữ cảnh. Mục đích là cho thấy cùng mẫu “một truy vấn so với nhiều ứng viên” trên một tác vụ quen thuộc.

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

Việc đầu tiên script này làm là tìm danh sách từ viết đúng của hệ điều hành. Vị trí của nó có thể thay đổi theo từng bản phân phối. Khi tìm thấy, ta nạp nó, bỏ xuống dòng và bắt đầu kiểm tra chính tả.

Kiểm tra chính tả trông rất giống việc so khớp trước đó. Với mỗi từ ứng viên, ta so nó với tất cả từ trong danh sách từ đúng, dùng `argmax()` để tìm ứng viên có điểm cao nhất, rồi thay từ bằng ứng viên đó. Có thể tăng tốc bằng vài tối ưu hóa, như không tìm khớp cho từ đã viết đúng, nhưng đây là demo và tối ưu hóa đó để lại làm bài tập cho người đọc.

Xem nó chạy thế nào!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Chi tiết

Khung hiện tại cung cấp:

- căn chỉnh Needleman-Wunsch chỉ tính điểm
- căn chỉnh Needleman-Wunsch có truy vết
- căn chỉnh Smith-Waterman chỉ tính điểm
- căn chỉnh Smith-Waterman có truy vết
- bố cục backend khớp với mẫu chuyên biệt hóa dùng trong `massive-speedup`
- phát hiện CPU/backend và điều phối backend phía Python

Ranh giới native chấp nhận:

- `bytes` với `bytes`
- `str` với `str`
- các chuỗi đối tượng Python bất biến và băm được
- đầu vào hỗn hợp chuỗi/đối tượng, trong đó phía `str` hoặc `bytes` được xử lý như một chuỗi

Cặp trực tiếp `bytes` với `str` sẽ ném `TypeError`.

Các triển khai hiện tại là kernel lập trình động tổng quát với bước tiền xử lý tuần tự hóa đầu vào Python thành luồng token 8, 16, 32 hoặc 64 bit. Backend chuyên biệt SIMD có thể thay thế các đơn vị dịch backend sau này mà không thay đổi API Python.

Hàm chỉ tính điểm trả về điểm số. Các biến thể chuẩn hóa trả về điểm giữa `0` và `1`. Hàm đường đi trả về đối tượng kết quả căn chỉnh chứa điểm, giá trị đã căn chỉnh, thao tác và, khi có, tóm tắt kiểu CIGAR.

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

Dùng `Scores(...).compare([...])` hoặc các hàm `*_scores()` cho tải công việc một truy vấn so với nhiều mục tiêu. Đường này chuẩn bị truy vấn/hồ sơ một lần và là API hiệu năng được ưu tiên cho so sánh văn bản tiếng Anh/tiếng Trung lặp lại.

Kết quả truy vết giữ nguyên kiểu đường nhanh theo cặp:

- `str` đầu vào trả về `str` đã căn chỉnh
- `bytes` đầu vào trả về `bytes` đã căn chỉnh
- đầu vào chuỗi/đối tượng trả về giá trị `tuple` đã căn chỉnh với khoảng trống `None`

Truyền `width=8`, `16`, `32` hoặc `64` để ép chiều rộng token/điểm nội bộ thay vì dùng chọn tự động.

Một số hàm hiển thị chuỗi CIGAR, viết tắt của “Concise Idiosyncratic Gapped Alignment Report”. CIGAR là ký hiệu thao tác căn chỉnh gọn dùng bởi công cụ SAM/BAM. Nếu muốn bản chính thức đầy đủ, xem [đặc tả SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Tối ưu hóa và benchmark

Hiệu năng của `stride-align` đã và sẽ tiếp tục được chú ý cẩn thận. Thư viện có tối ưu hóa SIMD cho nhiều đích phổ biến, gồm x86, Arm và LoongArch.

**LoongArch / Loongson.** Câu chuyện tối ưu hóa Loongson đặc biệt rõ ràng: trong ca benchmark đã kiểm tra với văn bản tiếng Anh, độ rộng điểm 16 bit và Smith-Waterman chỉ tính điểm, backend LASX nhanh hơn backend tổng quát 16 lần và nhanh hơn Parasail **22,4 lần**.

Nếu bạn là nhà nghiên cứu dùng máy chủ Loongson và hưởng lợi từ tăng tốc này, rất hoan nghênh trích dẫn, báo lỗi, ca benchmark và những món lưu niệm Trung Quốc nhỏ, rẻ. Trà, bookmark thư pháp, đồ trang trí cắt giấy, dây nút Trung Quốc, móc khóa gấu trúc và đồ trang trí bàn hình rồng nhỏ đều được chào đón. Xin đừng gửi thứ đắt tiền hoặc thứ cần giấy tờ hải quan.

Xem [benchmark đầy đủ](BENCHMARKS.md).

## Microbench native

Để phân tích hiệu năng không có frame Python hoặc dàn dựng benchmark, cấu hình bản dựng microbench x86 native:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` giữ các mô-đun nanobind chưa bị tước ký hiệu và thêm ký hiệu gỡ lỗi cùng con trỏ frame trong khi vẫn giữ `-O3`.

Đường cơ sở microbench native đã lưu nằm ở `benchmarks/x86_microbench_baseline.json`. Hãy xem nó như hàng rào cục bộ với ngưỡng lỏng, không phải SLA xuyên máy.


## Trích dẫn

Nếu bạn dùng phần mềm của tôi trong nghiên cứu, vui lòng trích dẫn tôi.

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
