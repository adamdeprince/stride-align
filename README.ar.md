# stride-align

**اللغات:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · **[العربية](README.ar.md)** · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` هي [مكتبة فائقة السرعة](BENCHMARK.md) تخبرك بمدى “تشابه” سلسلتين نصيتين. تفعل ذلك عبر تنفيذ خوارزميتي Smith-Waterman و Needleman-Wunsch. بدل محاضرة طويلة، سنتعلم بالتجربة. لنبدأ مباشرة بكيفية عملها.

## التثبيت

```bash
pip install stride-align
```

على أنظمة Loongson، ثبّت NumPy من توزيعة Linux الخاصة بك قبل تثبيت `stride-align`:

```bash
sudo apt install python3-numpy
pip install stride-align
```

أولاً، توضيح صغير: أنا لا أستخدم نصوصاً دينية هنا لدفع أي أجندة. في هذا العرض أحتاج إلى عدة وثائق كبيرة نسبياً من الملكية العامة، لها المعنى نفسه لكن بصياغات مختلفة. الكتاب المقدس يحدث فقط أنه يلائم هذا الشرط بشكل غريب جداً.

تخيّل أن لدينا جملتين؛ لنستخدم الجملة الأولى من Genesis لهذا المثال:

في American Standard Version لدينا: "In the beginning God created the heavens and the earth."

في King James Version لدينا: "In the beginning God created the heaven and the earth."

يمكننا رؤية الفرق بأعيننا: heavens مقابل heaven. لكن كيف نقيس هذا الفرق رقمياً؟ نستخدم هذا الجزء الصغير من الشيفرة:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

عند تشغيله يطبع:

```python
0.9907407407407407
```

الدرجات المطبّعة تقع بين `0` و `1`. الدرجة `1` تعني أن المدخلات متطابقة تماماً وفق نموذج التسجيل الافتراضي. الدرجات القريبة من `0` تعني أن المدخلات لا تشترك إلا بالقليل، رغم أن Smith-Waterman قد يظل قادراً على العثور على تطابقات محلية صغيرة داخل سلاسل غير مرتبطة عموماً.

الآن لنغيّر النص ونرى ماذا يحدث للدرجة.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

ويطبع Python:

```
0.12222222222222222
```

بدأت الفكرة تتضح؟ كلما كانت السلاسل أكثر تشابهاً ارتفعت الدرجة.

لننشئ مثالاً أكبر يعطينا إحساساً بأداء المكتبة. ستلاحظ على الأرجح أننا ننتقل بين Smith-Waterman و Needleman-Wunsch، وقد تتساءل متى تستخدم أياً منهما. استخدم Needleman-Wunsch عندما تريد مقارنة كامل المدخل بكامل المدخل. واستخدم Smith-Waterman عندما تريد العثور على أفضل منطقة مطابقة داخل مدخلات أكبر.

حسناً، لننتقل إلى شيفرة العرض. تحتاج إلى `requests` لهذا الجزء:

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

كيف نستخدم هذا؟ لنفترض أن لدينا آية عشوائية من الكتاب المقدس ونريد معرفة الفصل والآية التي جاءت منها. تقول `grep`؟ لا، يا للهول، لقد ارتكبنا خطأ. الآية التي لدينا من ترجمة مختلفة، لنقل Catholic Public Domain، وما على جهازنا هو King James Bible. المطابقة الحرفية للسلاسل في `grep` لن تنجح هنا. كيف نجد الفصل والآية؟ نبحث بالطبع عن السلسلة “الأقرب” أو “الأكثر تشابهاً” باستخدام `stride-align`.

في عرضنا، يتعامل الجزء الأول مع التنزيل والتخزين المؤقت. وضع الأشخاص الطيبون في [Open Bible](https://openbible.com) هذا النص في مكان يمكن الوصول إليه عبر HTTP، لكننا نريد احترام ميزانية تقنية المعلومات لديهم، لذلك نخزن ما ننزله مؤقتاً. هذه ببساطة مواطنة رقمية جيدة.

في الجزء التالي نحمّل كل الأسطر في قائمة. نزيل محارف السطر الجديد ونجعل كل شيء بأحرف صغيرة، لأننا لا نريد التدقيق المزعج في مسألة ضغط مفتاح Shift.

أخيراً، تجمع حلقة `while True:` سطراً من النص، على الأرجح الآية من النسخة Catholic من الكتاب المقدس التي نريد معرفة فصلها وآيتها، وتطابقها مع كل أسطر King James Bible باستخدام الصيغة الدُفعية من Needleman-Wunsch. ترجع مصفوفة من الدرجات. نستخدم `argmax()` للعثور على السطر الأعلى درجة، ثم نطبع السطر المرتبط بذلك الفهرس. لنجرب.

سأستخدم Jeremiah 4:28 من Catholic Bible؛ إنه في الواقع مختلف تماماً عن الآية نفسها في King James Bible. لنرَ ما يحدث…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

… ووجدناه! وبسرعة جيدة أيضاً.

## عرض آخر: التدقيق الإملائي

هذا مدقق إملائي تجريبي صغير، وليس أداة إنتاجية. يتجاهل علامات الترقيم، وحالة الأحرف، وتكرار الكلمات، والأسماء العلم، والسياق. الهدف هو عرض النمط نفسه: استعلام واحد مقابل كثير من المرشحين، على مهمة مألوفة.

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

أول ما يفعله هذا السكربت هو محاولة العثور على قائمة نظام التشغيل للكلمات المكتوبة بشكل صحيح. قد يختلف موقعها من توزيعة إلى أخرى. عندما نجدها، نحمّلها، ونزيل محارف السطر الجديد، ونبدأ عملية التدقيق الإملائي.

يشبه التدقيق الإملائي كثيراً المطابقة التي فعلناها سابقاً. لكل كلمة مرشحة، نطابقها مع كل الكلمات في قائمة الكلمات الصحيحة، ونستخدم `argmax()` للعثور على المرشح الأعلى درجة، ثم نستبدل الكلمة بذلك المرشح. يمكن تسريع هذا ببعض التحسينات، مثل عدم البحث عن مطابقة للكلمات المكتوبة أصلاً بشكل صحيح، لكن هذا عرض توضيحي وهذا التحسين متروك تمريناً للقارئ.

لنرَ كيف يعمل!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## التفاصيل

يوفر الهيكل الحالي ما يلي:

- محاذاة Needleman-Wunsch بدرجة فقط
- محاذاة Needleman-Wunsch مع تتبع عكسي
- محاذاة Smith-Waterman بدرجة فقط
- محاذاة Smith-Waterman مع تتبع عكسي
- تخطيط خلفية يطابق نمط التخصيص المستخدم في `massive-speedup`
- اكتشاف CPU/الخلفية وتوزيع الخلفية من جهة Python

تقبل الحدود الأصلية:

- `bytes` مقابل `bytes`
- `str` مقابل `str`
- تسلسلات من كائنات Python غير قابلة للتغيير وقابلة للتجزئة
- مدخلات مختلطة تسلسل/كائن حيث يعامل جانب `str` أو `bytes` كتسلسل

الأزواج المباشرة `bytes` مقابل `str` ترفع `TypeError`.

التنفيذات الحالية هي نوى عامة للبرمجة الديناميكية مع معالجة مسبقة تسلسل مدخلات Python إلى تدفقات رموز بعرض 8 أو 16 أو 32 أو 64 بت. يمكن للخلفيات المتخصصة لـ SIMD أن تستبدل لاحقاً وحدات ترجمة الخلفية من دون تغيير Python API.

الدوال التي تحسب الدرجة فقط ترجع درجات رقمية. النسخ المطبّعة ترجع درجات بين `0` و `1`. دوال المسار ترجع كائنات نتيجة محاذاة تحتوي على الدرجة، والقيم المحاذاة، والعمليات، وملخصات بأسلوب CIGAR عند توفرها.

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

استخدم `Scores(...).compare([...])` أو دوال `*_scores()` لأحمال العمل التي تقارن استعلاماً واحداً بكثير من الأهداف. هذا المسار يجهز الاستعلام/الملف التعريفي مرة واحدة، وهو API الأداء المفضل للمقارنات المتكررة بين النصوص الإنجليزية/الصينية.

تحافظ مخرجات التتبع العكسي على نوع المسار السريع المقترن:

- `str` كمدخلات ترجع `str` محاذاة
- `bytes` كمدخلات ترجع `bytes` محاذاة
- مدخلات التسلسل/الكائن ترجع قيماً `tuple` محاذاة مع فجوات `None`

مرر `width=8` أو `16` أو `32` أو `64` لفرض عرض الرمز/التسجيل الداخلي بدلاً من الاختيار التلقائي.

تكشف بعض الدوال سلاسل CIGAR، وهي اختصار لـ “Concise Idiosyncratic Gapped Alignment Report”. CIGAR هي ترميز مضغوط لعمليات المحاذاة تستخدمه أدوات SAM/BAM. إذا أردت النسخة الرسمية الكاملة، راجع [مواصفة SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## التحسينات والاختبارات المعيارية

حظيت قصة أداء `stride-align` وما زالت تحظى باهتمام دقيق. تتضمن المكتبة تحسينات SIMD لعدة أهداف شائعة، منها x86 و Arm و LoongArch.

**LoongArch / Loongson.** قصة تحسين Loongson معبرة جداً: في حالة الاختبار المعياري المحققة الخاصة بالنص الإنجليزي، وعرض الدرجة 16 بت، و Smith-Waterman الذي يحسب الدرجة فقط، يكون خلفية LASX أسرع 16 مرة من الخلفية العامة وأسرع **22.4 مرة** من Parasail.

إذا كنت باحثاً تستخدم خوادم Loongson وتستفيد من هذا التسريع، فالإحالات، وتقارير الأخطاء، وحالات الاختبار المعياري، والهدايا الصينية الصغيرة الرخيصة كلها محل تقدير. الشاي، وعلامات الكتب الخطية، وزخارف قص الورق، وتمائم العقد الصينية، وميداليات مفاتيح الباندا، وأجسام المكتب الصغيرة على شكل تنين كلها مرحب بها. من فضلك لا ترسل شيئاً غالياً أو شيئاً يتطلب أوراقاً جمركية.

راجع [الاختبارات المعيارية الكاملة](BENCHMARKS.md).

## اختبار ميكروي أصلي

لتحليل الأداء من دون إطارات Python أو تنسيق اختبارات معيارية، اضبط بناء اختبار ميكروي x86 أصلي:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` يبقي وحدات nanobind غير مجردة ويضيف رموز تصحيح ومؤشرات إطارات مع الحفاظ على `-O3`.

خط الأساس للاختبار الميكروي الأصلي المودع موجود في `benchmarks/x86_microbench_baseline.json`. عامله كحاجز محلي بعتبة مرنة، لا كاتفاقية مستوى خدمة بين آلات مختلفة.


## الاستشهادات

إذا استخدمت برمجيتي في بحثك، فيرجى الاستشهاد بي.

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
