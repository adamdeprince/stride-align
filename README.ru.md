# stride-align

**Языки:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · **[Русский](README.ru.md)** · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` — это [чертовски быстрая библиотека](BENCHMARK.md), которая показывает, насколько «похожи» две строки. Она делает это, реализуя алгоритмы Smith-Waterman и Needleman-Wunsch. Вместо лекции будем учиться на примерах. Сразу посмотрим, как это работает.

## Установка

```bash
pip install stride-align
```

На системах Loongson установите NumPy из своего дистрибутива Linux перед установкой `stride-align`:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Сначала небольшая оговорка: я использую здесь религиозные тексты не для продвижения какой-либо повестки. Для этой демонстрации мне нужно несколько довольно больших документов в общественном достоянии, которые имеют один смысл, но сформулированы по-разному. Библия просто пугающе хорошо подходит под это требование.

Представим, что у нас есть два предложения. Для этого возьмём первое предложение из Genesis:

В American Standard Version оно выглядит так: "In the beginning God created the heavens and the earth."

В King James Version оно выглядит так: "In the beginning God created the heaven and the earth."

На глаз видно различие: heavens против heaven. Но как это различие измерить? Используем небольшой фрагмент кода:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

При запуске он печатает:

```python
0.9907407407407407
```

Нормализованные оценки лежат между `0` и `1`. Оценка `1` означает, что входы точно совпадают при модели оценивания по умолчанию. Оценки около `0` означают, что у входов мало общего, хотя Smith-Waterman всё ещё может находить небольшие локальные совпадения внутри иначе несвязанных строк.

Теперь изменим текст и посмотрим, что произойдёт с оценкой.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

и Python печатает:

```
0.12222222222222222
```

Начинает проясняться? Чем больше похожи строки, тем выше оценка.

Построим пример побольше, чтобы почувствовать производительность библиотеки. Вы, вероятно, заметите, что мы переключаемся между Smith-Waterman и Needleman-Wunsch, и можете задуматься, когда что использовать. Используйте Needleman-Wunsch, когда хотите сравнить весь вход с другим входом целиком. Используйте Smith-Waterman, когда хотите найти лучшую совпадающую область внутри больших входов.

Хорошо, перейдём к демонстрационному коду. Для этой части нужен `requests`:

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

Как этим пользоваться? Допустим, у нас есть случайный библейский стих, и мы хотим узнать, из какой он главы и стиха. `grep`, говорите? О нет, мы ошиблись. Стих у нас из другой редакции, скажем Catholic Public Domain, а на компьютере лежит King James Bible. Точное строковое сопоставление `grep` здесь не сработает. Как найти главу и стих? Конечно, искать «ближайшую» или «самую похожую» строку с помощью `stride-align`.

В нашей демонстрации первая часть занимается загрузкой и кэшированием. Добрые люди из [Open Bible](https://openbible.com) разместили этот текст так, что он доступен по HTTP, но мы хотим уважать их ИТ-бюджет, поэтому кэшируем загруженное. Просто хорошее сетевое поведение.

В следующей части мы загружаем все строки в список. Убираем переводы строк и приводим всё к нижнему регистру, потому что не хотим придираться к тому, была ли нажата клавиша Shift.

Наконец, цикл `while True:` собирает строку текста — предположительно стих из Catholic-версии Библии, для которого мы хотим найти главу и стих, — и сопоставляет его со всеми строками King James Bible, используя пакетную форму Needleman-Wunsch. Он возвращает массив оценок. Мы используем `argmax()`, чтобы найти строку с лучшей оценкой, и печатаем строку, связанную с этим индексом. Попробуем.

Я возьму Jeremiah 4:28 из Catholic Bible — он действительно довольно сильно отличается от того же стиха в King James Bible. Посмотрим, что произойдёт…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

…и мы его нашли! И довольно быстро.

## Ещё одна демонстрация: проверка орфографии

Это игрушечная проверка орфографии, а не производственный инструмент. Она игнорирует пунктуацию, регистр, частоту слов, имена собственные и контекст. Смысл в том, чтобы показать тот же шаблон «один запрос против многих кандидатов» на знакомой задаче.

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

Первое, что делает этот скрипт, — пытается найти системный список правильно написанных слов. Его путь может отличаться от дистрибутива к дистрибутиву. Когда список найден, мы загружаем его, убираем переводы строк и начинаем проверку орфографии.

Проверка орфографии очень похожа на сопоставление, которое мы делали раньше. Для каждого слова-кандидата мы сравниваем его со всеми словами в списке правильных слов, используем `argmax()`, чтобы найти кандидата с наилучшей оценкой, и заменяем слово этим кандидатом. Можно ускорить это оптимизациями, например не искать совпадение для уже правильно написанных слов, но это демонстрация, и такая оптимизация оставлена читателю как упражнение.

Посмотрим, как это работает!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Подробности

Текущий каркас предоставляет:

- выравнивание Needleman-Wunsch только с оценкой
- выравнивание Needleman-Wunsch с обратной трассировкой
- выравнивание Smith-Waterman только с оценкой
- выравнивание Smith-Waterman с обратной трассировкой
- компоновку бэкендов, соответствующую шаблону специализации, используемому в `massive-speedup`
- обнаружение CPU/бэкенда и диспетчеризацию бэкенда на стороне Python

Нативная граница принимает:

- `bytes` против `bytes`
- `str` против `str`
- последовательности неизменяемых хешируемых объектов Python
- смешанные входы последовательность/объект, где сторона `str` или `bytes` трактуется как последовательность

Прямые пары `bytes` против `str` вызывают `TypeError`.

Текущие реализации — это обобщённые ядра динамического программирования с предварительной обработкой, которая сериализует входы Python в потоки токенов шириной 8, 16, 32 или 64 бита. SIMD-специализированные бэкенды смогут позже заменить единицы трансляции бэкенда без изменения Python API.

Функции только с оценкой возвращают числовые оценки. Нормализованные варианты возвращают оценки между `0` и `1`. Функции пути возвращают объекты результата выравнивания, содержащие оценку, выровненные значения, операции и, когда доступно, сводки в стиле CIGAR.

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

Используйте `Scores(...).compare([...])` или функции `*_scores()` для нагрузок «один запрос против многих целей». Этот путь один раз подготавливает запрос/профиль и является предпочтительным высокопроизводительным API для повторяющихся сравнений английского/китайского текста.

Выводы обратной трассировки сохраняют парный тип быстрого пути:

- `str` на входе возвращают выровненные `str`
- `bytes` на входе возвращают выровненные `bytes`
- входы последовательность/объект возвращают выровненные значения `tuple` с пропусками `None`

Передайте `width=8`, `16`, `32` или `64`, чтобы принудительно задать внутреннюю ширину токенов/оценивания вместо автоматического выбора.

Некоторые функции предоставляют строки CIGAR, сокращение от “Concise Idiosyncratic Gapped Alignment Report”. CIGAR — это компактная запись операций выравнивания, используемая инструментами SAM/BAM. Если нужна полная формальная версия, см. [спецификацию SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Оптимизации и тесты производительности

Производительности `stride-align` уделялось и продолжает уделяться серьёзное внимание. Библиотека включает SIMD-оптимизации для ряда распространённых целей, включая x86, Arm и LoongArch.

**LoongArch / Loongson.** История оптимизации Loongson особенно показательна: в проверенном бенчмарк-сценарии с английским текстом, 16-битной шириной оценки и Smith-Waterman только для вычисления оценки бэкенд LASX в 16 раз быстрее универсального бэкенда и в **22,4 раза** быстрее Parasail.

Если вы исследователь, используете серверы Loongson и получаете пользу от этого ускорения, приветствуются цитирования, отчёты об ошибках, тестовые случаи и маленькие недорогие китайские сувениры. Чай, каллиграфические закладки, украшения из вырезанной бумаги, китайские узелки, брелоки с пандой и маленькие настольные драконы — всё это приветствуется. Пожалуйста, не присылайте ничего дорогого и ничего, что требует таможенных документов.

См. [полные тесты производительности](BENCHMARKS.md).

## Нативный микробенчмарк

Для профилирования производительности без Python-фреймов и оркестрации тестов настройте нативную x86-сборку микробенчмарка:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` оставляет модули nanobind неразобранными, добавляет отладочные символы и указатели кадров, сохраняя `-O3`.

Зафиксированная базовая линия нативного микробенчмарка находится в `benchmarks/x86_microbench_baseline.json`. Рассматривайте её как локальный ограничитель с мягким порогом, а не как SLA между машинами.


## Цитирование

Если вы используете моё программное обеспечение в исследовании, пожалуйста, процитируйте меня.

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
