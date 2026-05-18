# stride-align

**भाषाएँ:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · **[हिन्दी](README.hi.md)** · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` एक [बेहद तेज़ लाइब्रेरी](BENCHMARK.md) है जो बताती है कि दो स्ट्रिंग कितनी “मिलती-जुलती” हैं। यह Smith-Waterman और Needleman-Wunsch एल्गोरिदम लागू करके ऐसा करती है। लंबा व्याख्यान देने के बजाय हम काम करके सीखेंगे। चलिए सीधे देखते हैं कि यह कैसे काम करती है।

## इंस्टॉलेशन

```bash
pip install stride-align
```

Loongson सिस्टम पर `stride-align` इंस्टॉल करने से पहले अपने Linux वितरण से NumPy इंस्टॉल करें:

```bash
sudo apt install python3-numpy
pip install stride-align
```

पहले एक बात साफ कर दूँ: यहाँ धार्मिक ग्रंथों का उपयोग किसी एजेंडा को बढ़ाने के लिए नहीं है। इस डेमो के लिए मुझे ऐसे कई काफ़ी बड़े public domain दस्तावेज़ चाहिए जिनका अर्थ समान हो, लेकिन भाषा अलग-अलग हो। बाइबल बस इस ज़रूरत पर अजीब तरह से अच्छी बैठती है।

मान लीजिए हमारे पास दो वाक्य हैं; इसके लिए Genesis का पहला वाक्य लेते हैं:

American Standard Version में हमारे पास है: "In the beginning God created the heavens and the earth."

King James Version में हमारे पास है: "In the beginning God created the heaven and the earth."

आँख से हमें अंतर दिखता है: heavens बनाम heaven। लेकिन इस अंतर को संख्या में कैसे मापें? इस छोटे से कोड का उपयोग करेंगे:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

इसे चलाने पर यह छापता है:

```python
0.9907407407407407
```

सामान्यीकृत स्कोर `0` और `1` के बीच होते हैं। `1` का स्कोर मतलब है कि डिफ़ॉल्ट scoring मॉडल में इनपुट बिल्कुल मिलते हैं। `0` के पास स्कोर का अर्थ है कि इनपुट में बहुत कम समानता है, हालाँकि Smith-Waterman असंबंधित स्ट्रिंगों के अंदर भी छोटे स्थानीय मिलान खोज सकता है।

अब टेक्स्ट बदलते हैं और देखते हैं स्कोर के साथ क्या होता है।

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

और Python छापता है:

```
0.12222222222222222
```

अब बात समझ आ रही है? स्ट्रिंग जितनी अधिक समान होंगी, स्कोर उतना अधिक होगा।

अब एक बड़ा उदाहरण बनाते हैं, जिससे लाइब्रेरी के प्रदर्शन का अंदाज़ा मिले। आप शायद देखेंगे कि हम Smith-Waterman और Needleman-Wunsch के बीच बदलते हैं, और सोचेंगे कि कब कौन सा इस्तेमाल करना चाहिए। जब आप पूरी इनपुट की तुलना पूरी इनपुट से करना चाहते हैं, Needleman-Wunsch इस्तेमाल करें। जब आप बड़े इनपुट के अंदर सबसे अच्छा मेल खाने वाला क्षेत्र ढूँढना चाहते हैं, Smith-Waterman इस्तेमाल करें।

ठीक है, अब डेमो कोड पर चलते हैं। इस भाग के लिए `requests` चाहिए:

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

इसका उपयोग कैसे करें? मान लीजिए हमारे पास बाइबल की कोई यादृच्छिक पंक्ति है और हम जानना चाहते हैं कि वह किस अध्याय और पद से आती है। आप कहेंगे `grep`? अरे नहीं, हमने गलती कर दी। हमारे पास जो पद है वह किसी दूसरी translation, जैसे Catholic Public Domain, से है और कंप्यूटर पर King James Bible है। `grep` का exact string matching यहाँ काम नहीं करेगा। अध्याय और पद कैसे खोजें? ज़ाहिर है, `stride-align` से “सबसे नज़दीकी” या “सबसे समान” स्ट्रिंग खोजेंगे।

हमारे डेमो में पहला भाग डाउनलोड और कैशिंग संभालता है। [Open Bible](https://openbible.com) के अच्छे लोगों ने यह टेक्स्ट HTTP से उपलब्ध कराया है, लेकिन हमें उनका IT बजट सम्मान करना है, इसलिए हम डाउनलोड की गई चीज़ कैश करते हैं। यह बस अच्छी नागरिकता है।

अगले भाग में हम सभी पंक्तियों को एक सूची में लोड करते हैं। हम newline हटाते हैं और सब कुछ lowercase करते हैं क्योंकि Shift कुंजी पकड़े हुए थे या नहीं, इस पर झंझट नहीं करना चाहते।

अंत में `while True:` लूप टेक्स्ट की एक पंक्ति लेता है, शायद Catholic Bible वाला वह पद जिसका अध्याय और पद हमें पता करना है, और Needleman-Wunsch के batch रूप से उसे King James Bible की सभी पंक्तियों से मिलाता है। यह scores की array लौटाता है। हम `argmax()` से सबसे अच्छा स्कोर वाली पंक्ति ढूँढते हैं और उस index से जुड़ी पंक्ति छापते हैं। चलिए कोशिश करें।

मैं Catholic Bible से Jeremiah 4:28 इस्तेमाल करूँगा; यह King James Bible में उसी पद से वास्तव में काफी अलग है। देखते हैं क्या होता है…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

…और हमने उसे खोज लिया! वह भी काफी जल्दी।

## एक और डेमो: वर्तनी जाँच

यह खिलौना-स्तर का spell checker है, production वाला नहीं। यह punctuation, capitalization, word frequency, proper nouns और context को नज़रअंदाज़ करता है। मकसद है एक परिचित काम पर वही “एक query बनाम कई candidates” वाला pattern दिखाना।

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

यह script सबसे पहले हमारे operating system की सही वर्तनी वाले शब्दों की सूची खोजने की कोशिश करता है। उसका स्थान distribution के हिसाब से बदल सकता है। मिल जाने पर हम उसे लोड करते हैं, newlines हटाते हैं और वर्तनी जाँच शुरू करते हैं।

वर्तनी जाँच पहले किए गए matching जैसी ही दिखती है। हर candidate शब्द के लिए हम उसे सही वर्तनी वाले शब्दों की सूची के सभी शब्दों से मिलाते हैं, `argmax()` से सबसे अच्छे score वाला candidate खोजते हैं, और शब्द को उस candidate से बदल देते हैं। हम कुछ optimizations से इसे तेज़ कर सकते हैं, जैसे सही लिखे शब्दों के लिए match न खोजना, लेकिन यह demo है और वह optimization पाठक के लिए अभ्यास है।

देखें यह कैसे चलता है!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## विवरण

वर्तमान scaffold यह देता है:

- केवल स्कोर वाला Needleman-Wunsch संरेखण
- ट्रेसबैक के साथ Needleman-Wunsch संरेखण
- केवल स्कोर वाला Smith-Waterman संरेखण
- ट्रेसबैक के साथ Smith-Waterman संरेखण
- ऐसा बैकएंड विन्यास जो `massive-speedup` में उपयोग किए गए विशेषीकरण पैटर्न से मेल खाता है
- CPU/बैकएंड पहचान और Python-पक्ष बैकएंड प्रेषण

नेटिव सीमा स्वीकार करती है:

- `bytes` के विरुद्ध `bytes`
- `str` के विरुद्ध `str`
- अपरिवर्तनीय और हैश योग्य Python ऑब्जेक्टों के अनुक्रम
- मिश्रित अनुक्रम/ऑब्जेक्ट इनपुट जहाँ `str` या `bytes` वाला पक्ष अनुक्रम की तरह माना जाता है

सीधे `bytes` बनाम `str` जोड़े `TypeError` उठाते हैं।

वर्तमान कार्यान्वयन प्री-प्रोसेसिंग वाले सामान्य गतिशील-प्रोग्रामिंग कर्नेल हैं, जो Python इनपुट को 8, 16, 32, या 64-bit टोकन धाराओं में क्रमबद्ध करते हैं। SIMD-विशेषीकृत बैकएंड बाद में बैकएंड अनुवाद इकाइयों को Python API बदले बिना बदल सकते हैं।

केवल-स्कोर फ़ंक्शन संख्यात्मक स्कोर लौटाते हैं। सामान्यीकृत रूप `0` और `1` के बीच स्कोर लौटाते हैं। पथ फ़ंक्शन संरेखण-परिणाम ऑब्जेक्ट लौटाते हैं जिनमें स्कोर, संरेखित मान, संचालन, और उपलब्ध होने पर CIGAR-शैली सारांश होती हैं।

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

एक प्रश्न को कई लक्ष्यों से स्कोर करने वाले कार्यभार के लिए `Scores(...).compare([...])` या `*_scores()` फ़ंक्शनों का उपयोग करें। यह पथ प्रश्न/प्रोफ़ाइल को एक बार तैयार करता है और बार-बार की अंग्रेज़ी/चीनी पाठ तुलना के लिए पसंदीदा प्रदर्शन API है।

ट्रेसबैक आउटपुट युग्मित तेज़-पथ प्रकार को सुरक्षित रखते हैं:

- `str` इनपुट संरेखित `str` लौटाते हैं
- `bytes` इनपुट संरेखित `bytes` लौटाते हैं
- अनुक्रम/ऑब्जेक्ट इनपुट `None` अंतरालों वाले संरेखित `tuple` मान लौटाते हैं

स्वचालित चयन की जगह आंतरिक टोकन/स्कोरिंग चौड़ाई मजबूर करने के लिए `width=8`, `16`, `32`, या `64` पास करें।

कुछ फ़ंक्शन CIGAR स्ट्रिंग देते हैं, जो “Concise Idiosyncratic Gapped Alignment Report” का संक्षिप्त रूप है। CIGAR वह सघन संरेखण-कार्यवाही संकेत-लेखन है जो SAM/BAM औज़ारों में उपयोग होती है। पूरी औपचारिक संस्करण के लिए [SAM विनिर्देश](https://samtools.github.io/hts-specs/SAMv1.pdf) देखें।

## अनुकूलन और बेंचमार्क

`stride-align` की प्रदर्शन कहानी पर सावधानी से ध्यान दिया गया है और आगे भी दिया जाएगा। लाइब्रेरी x86, Arm और LoongArch सहित कई सामान्य लक्ष्यों के लिए SIMD अनुकूलन शामिल करती है।

**LoongArch / Loongson.** Loongson अनुकूलन कहानी खास तौर पर साफ़ है: जाँचे गए बेंचमार्क मामले में, अंग्रेज़ी पाठ, 16-बिट स्कोर चौड़ाई, और केवल-स्कोर Smith-Waterman के लिए LASX बैकएंड सामान्य बैकएंड से 16 गुना तेज़ और Parasail से **22.4 गुना** तेज़ है।

यदि आप Loongson सर्वर इस्तेमाल करने वाले शोधकर्ता हैं और इस गति-वृद्धि से लाभ उठा रहे हैं, तो उद्धरण, बग रिपोर्ट, बेंचमार्क मामले और छोटे सस्ते चीनी स्मृति-चिह्न सराहे जाएँगे। चाय, सुलेख बुकमार्क, कागज़-कटाई सजावट, चीनी गाँठ लटकन, पांडा चाबी-छल्ले और छोटे ड्रैगन डेस्क सजावटी सामान सब स्वागत योग्य हैं। कृपया कोई महँगी चीज़ या सीमा-शुल्क कागज़ी कार्रवाई वाली चीज़ न भेजें।

[पूरे बेंचमार्क](BENCHMARKS.md) देखें।

## नेटिव माइक्रोबेंच

Python फ़्रेम या बेंचमार्क संयोजन के बिना प्रदर्शन प्रोफ़ाइलिंग के लिए नेटिव x86 माइक्रोबेंच बिल्ड कॉन्फ़िगर करें:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` nanobind मॉड्यूल को अनस्ट्रिप्ड रखता है और debug symbols तथा frame pointers जोड़ता है, जबकि `-O3` बनाए रखता है।

जमा की गई नेटिव माइक्रोबेंच आधार-रेखा `बेंचमार्क/x86_microbench_baseline.json` पर है। इसे ढीली सीमा वाला स्थानीय सुरक्षा-घेरा मानें, मशीन-पार SLA नहीं।


## Citations

यदि आप मेरे सॉफ़्टवेयर को अपने शोध में उपयोग करते हैं, कृपया मुझे उद्धृत करें।

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
