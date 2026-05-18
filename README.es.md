# stride-align

**Idiomas:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · **[Español](README.es.md)** · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` es una biblioteca que te dice qué tan “parecidas” son dos cadenas. Lo hace implementando los algoritmos Smith-Waterman y Needleman-Wunsch. En lugar de darte una clase, vamos a aprender haciendo. Entremos directamente en cómo funciona.

## Instalación

```bash
pip install stride-align
```

En sistemas Loongson, instala NumPy desde tu distribución Linux antes de instalar `stride-align`:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Primero, una aclaración: no estoy usando textos religiosos aquí para promover una agenda. Para esta demo necesito varios documentos medianamente grandes, de dominio público, que tengan el mismo significado pero estén redactados de forma distinta. La Biblia simplemente encaja con ese requisito de manera casi ridículamente perfecta.

Imagina que tenemos dos frases; usemos para esto la primera frase del Génesis:

En la American Standard Version tenemos: "In the beginning God created the heavens and the earth."

En la King James Version tenemos: "In the beginning God created the heaven and the earth."

A simple vista vemos una diferencia: heavens frente a heaven. Pero ¿cómo cuantificamos esa diferencia? Usamos este pequeño fragmento de código:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Al ejecutarlo imprime:

```python
0.9907407407407407
```

Los puntajes normalizados están entre `0` y `1`. Un puntaje de `1` significa que las entradas coinciden exactamente bajo el modelo de puntuación predeterminado. Los puntajes cercanos a `0` significan que las entradas tienen poco en común, aunque Smith-Waterman todavía puede encontrar pequeñas coincidencias locales dentro de cadenas que, por lo demás, no están relacionadas.

Ahora cambiemos el texto y veamos qué ocurre con el puntaje.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

y Python imprime:

```
0.12222222222222222
```

¿Ya se entiende la idea? Cuanto más parecidas sean las cadenas, más alto será el puntaje.

Construyamos un ejemplo más grande, algo que nos dé una idea del rendimiento de la biblioteca. Probablemente notarás que cambiamos entre Smith-Waterman y Needleman-Wunsch, y quizá te preguntes cuál usar y cuándo. Usa Needleman-Wunsch cuando quieras comparar toda la entrada contra toda la entrada. Usa Smith-Waterman cuando quieras encontrar la región de mejor coincidencia dentro de entradas más grandes.

Bien, pasemos al código de la demo. Necesitas `requests` para esta parte:

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

¿Cómo usamos esto? Supón que tenemos un versículo bíblico al azar y queremos saber de qué capítulo y versículo viene. ¿Dices `grep`? Ay, no, cometimos un error. El versículo que tenemos viene de otra traducción, digamos Catholic Public Domain, y en nuestra computadora tenemos la King James Bible. La coincidencia exacta de cadenas de `grep` no funcionará aquí. ¿Cómo encontramos el capítulo y el versículo? Buscamos la cadena “más cercana” o “más parecida” usando `stride-align`, por supuesto.

En nuestra demo, la primera parte se ocupa de descargar y guardar en caché. La buena gente de [Open Bible](https://openbible.com) puso este texto donde se puede acceder por HTTP, pero queremos respetar su presupuesto de TI, así que guardamos en caché lo que descargamos. Es simple buena ciudadanía.

En la siguiente parte cargamos todas las líneas en una lista. Quitamos saltos de línea y convertimos todo a minúsculas porque no queremos ponernos quisquillosos con si alguien estaba presionando Shift.

Por último, el bucle `while True:` recoge una línea de texto, presumiblemente el versículo de la versión Catholic de la Biblia cuyo capítulo y versículo queremos buscar, y lo compara contra todas las líneas de la King James Bible usando la forma por lotes de Needleman-Wunsch. Devuelve un arreglo de puntajes. Usamos `argmax()` para encontrar la línea con mejor puntaje y luego imprimimos la línea asociada con ese índice. Probémoslo.

Voy a usar Jeremiah 4:28 de la Catholic Bible; en realidad es bastante distinto del mismo versículo en la King James Bible. Veamos qué pasa…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

… ¡y lo encontramos! Y bastante rápido.

## Otra demo: corrección ortográfica

Este es un corrector ortográfico de juguete, no uno de producción. Ignora puntuación, mayúsculas, frecuencia de palabras, nombres propios y contexto. El objetivo es mostrar el mismo patrón de “una consulta contra muchos candidatos” en una tarea conocida.

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

Lo primero que hace este script es intentar encontrar la lista de palabras correctamente escritas del sistema operativo. Su ubicación puede variar de una distribución a otra. Una vez que la encontramos, la cargamos, quitamos los saltos de línea y comenzamos la corrección ortográfica.

La corrección se parece mucho a la coincidencia que hicimos antes. Para cada palabra candidata, la comparamos contra todas las palabras de nuestra lista de palabras correctas, usamos `argmax()` para encontrar la candidata con mayor puntaje y reemplazamos la palabra por esa candidata. Podríamos acelerarlo con optimizaciones, como no buscar coincidencia para palabras que ya están bien escritas, pero esto es una demo y esa optimización queda como ejercicio para el lector.

¡Veamos cómo funciona!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Detalles

La estructura actual proporciona:

- alineación Needleman-Wunsch solo con puntaje
- alineación Needleman-Wunsch con retrotrazado
- alineación Smith-Waterman solo con puntaje
- alineación Smith-Waterman con retrotrazado
- una disposición de backends que coincide con el patrón de especialización usado en `massive-speedup`
- detección de CPU/backend y despacho de backend del lado de Python

La frontera nativa acepta:

- `bytes` contra `bytes`
- `str` contra `str`
- secuencias de objetos Python inmutables y hashables
- entradas mixtas de secuencia/objeto donde el lado `str` o `bytes` se trata como una secuencia

Los pares directos `bytes` frente a `str` lanzan `TypeError`.

Las implementaciones actuales son núcleos genéricos de programación dinámica con preprocesamiento que serializa entradas Python en flujos de tokens de 8, 16, 32 o 64 bits. Los backends especializados para SIMD podrán reemplazar más adelante las unidades de traducción del backend sin cambiar la API de Python.

Las funciones de solo puntaje devuelven puntajes numéricos. Las variantes normalizadas devuelven puntajes entre `0` y `1`. Las funciones de ruta devuelven objetos de resultado de alineación que contienen el puntaje, los valores alineados, las operaciones y, cuando está disponible, resúmenes estilo CIGAR.

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

Usa `Scores(...).compare([...])` o las funciones `*_scores()` para cargas de trabajo de una consulta contra muchos objetivos. Esa ruta prepara la consulta/perfil una sola vez y es la API de rendimiento preferida para comparaciones repetidas de texto en inglés/chino.

Las salidas de retrotrazado conservan el tipo de vía rápida emparejada:

- `str` de entrada devuelven `str` alineados
- `bytes` de entrada devuelven `bytes` alineados
- las entradas de secuencia/objeto devuelven valores `tuple` alineados con huecos `None`

Pasa `width=8`, `16`, `32` o `64` para forzar el ancho interno de token/puntuación en lugar de usar la selección automática.

Algunas funciones exponen cadenas CIGAR, abreviatura de “Concise Idiosyncratic Gapped Alignment Report”. CIGAR es la notación compacta de operaciones de alineación usada por herramientas SAM/BAM. Si quieres la versión formal completa, consulta la [especificación SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Optimizaciones y benchmarks

Se ha prestado, y se sigue prestando, mucha atención a la historia de rendimiento de `stride-align`. La biblioteca incluye optimización SIMD para varios objetivos comunes, incluidos x86, Arm y LoongArch.

**LoongArch / Loongson.** La historia de optimización para Loongson es especialmente reveladora: para cálculos de solo puntaje en el benchmark verificado, el backend LASX es 16 veces más rápido que el backend genérico y **22,3 veces** más rápido que Parasail.

Si eres investigador, usas servidores Loongson y te beneficias de esta aceleración, se agradecen citas, informes de errores, casos de benchmark y pequeños recuerdos chinos baratos. Té, marcadores de caligrafía, adornos de papel recortado, amuletos de nudo chino, llaveros de panda y pequeños objetos de escritorio con forma de dragón son bienvenidos. Por favor no envíes nada caro ni nada que requiera trámites de aduana.

Consulta los [benchmarks completos](BENCHMARKS.md).

## Microbenchmark nativo

Para perfilar rendimiento sin marcos de Python ni orquestación de benchmarks, configura una compilación nativa de microbenchmark x86:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` mantiene los módulos nanobind sin despojar y agrega símbolos de depuración más punteros de marco, conservando `-O3`.

La línea base de microbenchmark nativo registrada vive en `benchmarks/x86_microbench_baseline.json`. Trátala como una barrera local con umbral laxo, no como un SLA entre máquinas.


## Citas

Si usas mi software en tu investigación, por favor cítame.

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
