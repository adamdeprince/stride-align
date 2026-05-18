# stride-align

**Sprachen:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · **[Deutsch](README.de.md)** · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` ist eine Bibliothek, die dir sagt, wie „ähnlich“ zwei Zeichenketten sind. Sie tut das, indem sie die Algorithmen Smith-Waterman und Needleman-Wunsch implementiert. Statt einer Vorlesung lernen wir durch Ausprobieren. Springen wir direkt hinein. 

## Installation

```bash
pip install stride-align
```

Auf Loongson-Systemen installiere NumPy aus deiner Linux-Distribution, bevor du `stride-align` installierst:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Zuerst ein Hinweis: Ich benutze religiöse Texte hier nicht, um irgendeine Agenda zu schieben. Für diese Demo brauche ich mehrere recht große gemeinfreie Dokumente, die dieselbe Bedeutung haben, aber anders formuliert sind. Die Bibel passt für diese Demo nur erschreckend gut.

Stell dir vor, wir haben zwei Sätze. Nehmen wir den ersten Satz aus der Genesis:

In der American Standard Version steht: "In the beginning God created the heavens and the earth."

In der King James Version steht: "In the beginning God created the heaven and the earth."

Mit bloßem Auge sehen wir den Unterschied: heavens gegen heaven. Aber wie messen wir diesen Unterschied? Mit diesem kleinen Stück Code:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Wenn wir das ausführen, gibt es aus:

```python
0.9907407407407407
```

Normalisierte Werte liegen zwischen `0` und `1`. Ein Wert von `1` bedeutet, dass die Eingaben unter dem Standard-Bewertungsmodell exakt übereinstimmen. Werte nahe `0` bedeuten, dass die Eingaben wenig gemeinsam haben, auch wenn Smith-Waterman in ansonsten unzusammenhängenden Zeichenketten noch kleine lokale Treffer finden kann.

Ändern wir nun den Text und schauen, was mit dem Wert passiert.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

und Python gibt aus:

```
0.12222222222222222
```

Kommt die Idee an? Je ähnlicher die Zeichenketten, desto höher der Wert.

Bauen wir ein größeres Beispiel, damit man ein Gefühl für die Leistung der Bibliothek bekommt. Dir fällt vermutlich auf, dass wir zwischen Smith-Waterman und Needleman-Wunsch wechseln, und du fragst dich vielleicht, wann man was benutzt. Verwende Needleman-Wunsch, wenn du die gesamte Eingabe mit der gesamten Eingabe vergleichen willst. Verwende Smith-Waterman, wenn du den am besten passenden Bereich innerhalb größerer Eingaben finden willst.

Gut, weiter zum Democode. Für diesen Teil der Demo brauchst du `requests`:

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

Wie können wir das nun nutzen? Angenommen, wir haben einen zufälligen Bibelvers und wollen wissen, aus welchem Kapitel und Vers er stammt. `grep`, sagst du? Ach herrje, nein, wir haben einen Fehler gemacht. Der Vers liegt in einer anderen Übersetzung vor, sagen wir Catholic Public Domain, und auf unserem Rechner haben wir die King James Bible. Die exakte Zeichenkettensuche von `grep` funktioniert hier nicht. Wie finden wir Kapitel und Vers? Natürlich suchen wir mit `stride-align` nach der „nächsten“ oder „ähnlichsten“ Zeichenkette.

Im ersten Teil der Demo geht es um Herunterladen und Zwischenspeichern. Die netten Leute von [Open Bible](https://openbible.com) haben diesen Text per HTTP erreichbar abgelegt, aber wir wollen ihr IT-Budget respektieren, also speichern wir heruntergeladene Daten lokal zwischen. Das ist einfach anständiges Netzbürgerverhalten.

Im nächsten Teil laden wir alle Zeilen in eine Liste. Wir entfernen Zeilenumbrüche und machen alles kleingeschrieben, weil wir nicht pedantisch darüber werden wollen, ob jemand die Umschalttaste gedrückt hat.

Zuletzt sammelt die Schleife `while True:` eine Textzeile ein, vermutlich den Vers aus der Catholic-Version der Bibel, für den wir Kapitel und Vers nachschlagen wollen, und gleicht ihn mit allen Zeilen der King James Bible über die Batch-Form von Needleman-Wunsch ab. Sie liefert ein Array von Werten zurück. Mit `argmax()` finden wir die Zeile mit dem höchsten Wert und geben dann die zu diesem Index gehörende Zeile aus. Probieren wir es.

Ich nehme Jeremiah 4:28 aus der Catholic Bible. Der Vers unterscheidet sich tatsächlich ziemlich stark vom selben Vers in der King James Bible. Schauen wir, was passiert …

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

… und wir haben ihn gefunden! Und ziemlich schnell noch dazu.

## Noch eine Demo: Rechtschreibprüfung

Das ist eine Spielzeug-Rechtschreibprüfung, keine produktionsreife. Sie ignoriert Zeichensetzung, Groß-/Kleinschreibung, Worthäufigkeit, Eigennamen und Kontext. Der Punkt ist, dasselbe Muster „eine Anfrage gegen viele Kandidaten“ an einer vertrauten Aufgabe zu zeigen.

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

Als Erstes versucht dieses Skript, die Liste korrekt geschriebener Wörter unseres Betriebssystems zu finden. Ihr Ort kann je nach Distribution variieren. Sobald wir sie gefunden haben, laden wir sie, entfernen Zeilenumbrüche und beginnen mit der Rechtschreibprüfung.

Die Rechtschreibprüfung sieht sehr ähnlich aus wie der Abgleich davor. Für jedes Kandidatenwort gleichen wir es gegen alle Wörter in unserer Liste korrekt geschriebener Wörter ab, verwenden `argmax()`, um den Kandidaten mit dem höchsten Wert zu finden, und ersetzen das Wort durch diesen Kandidaten. Man könnte das mit Optimierungen beschleunigen, zum Beispiel indem man für bereits richtig geschriebene Wörter gar nicht sucht, aber dies ist eine Demo, und diese Optimierung bleibt dem Leser als Übung überlassen.

Schauen wir, wie es funktioniert!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Details

Das aktuelle Grundgerüst stellt bereit:

- Needleman-Wunsch-Abgleich nur mit Bewertung
- Needleman-Wunsch-Abgleich mit Rückverfolgung
- Smith-Waterman-Abgleich nur mit Bewertung
- Smith-Waterman-Abgleich mit Rückverfolgung
- eine Backend-Anordnung, die dem in `massive-speedup` verwendeten Spezialisierungsmuster entspricht
- CPU-/Backend-Erkennung und Backend-Verteilung auf der Python-Seite

Die native Grenze akzeptiert:

- `bytes` gegen `bytes`
- `str` gegen `str`
- Sequenzen unveränderlicher, hashbarer Python-Objekte
- gemischte Sequenz-/Objekt-Eingaben, bei denen eine `str`- oder `bytes`-Seite als Sequenz behandelt wird

Direkte Paare aus `bytes` und `str` lösen `TypeError` aus.

Die aktuellen Implementierungen sind generische dynamische-Programmierungs-Kerne mit Vorverarbeitung, die Python-Eingaben in 8-, 16-, 32- oder 64-Bit-Tokenströme serialisiert. SIMD-spezialisierte Backends können später die Backend-Übersetzungseinheiten ersetzen, ohne die Python-API zu ändern.

Funktionen, die nur eine Bewertung berechnen, geben numerische Werte zurück. Die normalisierten Varianten geben Werte zwischen `0` und `1` zurück. Pfadfunktionen geben Abgleich-Ergebnisobjekte zurück, die Bewertung, ausgerichtete Werte, Operationen und, falls verfügbar, CIGAR-artige Zusammenfassungen enthalten.

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

Nutze `Scores(...).compare([...])` oder die `*_scores()`-Funktionen für Arbeitslasten mit einer Anfrage gegen viele Zieltexte. Dieser Pfad bereitet Anfrage/Profil einmal vor und ist die bevorzugte Leistungs-API für wiederholte Englisch/Chinesisch-Textvergleiche.

Rückverfolgungsausgaben behalten den gepaarten Schnellpfad-Typ bei:

- `str` Eingaben geben ausgerichtete `str` zurück
- `bytes` Eingaben geben ausgerichtete `bytes` zurück
- Sequenz-/Objekt-Eingaben geben ausgerichtete `tuple`-Werte mit `None`-Lücken zurück

Gib `width=8`, `16`, `32` oder `64` an, um die interne Token-/Bewertungsbreite zu erzwingen, statt die automatische Auswahl zu verwenden.

Einige Funktionen stellen CIGAR-Zeichenketten bereit, kurz für “Concise Idiosyncratic Gapped Alignment Report”. CIGAR ist die kompakte Notation für Abgleichoperationen, die SAM/BAM-Werkzeuge verwenden. Wenn du die vollständige formale Version willst, siehe die [SAM-Spezifikation](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Optimierungen und Benchmarks

Der Leistungsgeschichte von `stride-align` wurde und wird weiterhin sorgfältige Aufmerksamkeit gewidmet. Die Bibliothek enthält SIMD-Optimierungen für verschiedene gängige Ziele, darunter x86, Arm und LoongArch.

**LoongArch / Loongson.** Die Loongson-Optimierung ist besonders aussagekräftig: Bei reinen Bewertungsberechnungen im geprüften Benchmark ist das LASX-Backend 16-mal schneller als das generische Backend und **22,3-mal** schneller als Parasail.

Wenn du als Forscher Loongson-Server verwendest und von dieser Beschleunigung profitierst, sind Zitate, Fehlerberichte, Benchmark-Fälle und kleine, günstige chinesische Souvenirs willkommen. Tee, Kalligrafie-Lesezeichen, Scherenschnitt-Ornamente, chinesische Knotenanhänger, Panda-Schlüsselanhänger und kleine Drachen-Schreibtischobjekte sind alle gern gesehen. Bitte nichts Teures schicken und nichts, wofür Zollpapiere nötig sind.

Siehe [vollständige Benchmarks](BENCHMARKS.md).

## Native Mikrobenchmarks

Für Leistungsprofiling ohne Python-Frames oder Benchmark-Orchestrierung richte einen nativen x86-Mikrobenchmark-Build ein:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` lässt nanobind-Module ungestrippt und fügt Debugsymbole plus Framepointer hinzu, während `-O3` erhalten bleibt.

Die eingecheckte native Mikrobenchmark-Basislinie liegt in `benchmarks/x86_microbench_baseline.json`. Behandle sie als lokale Leitplanke mit lockerem Schwellwert, nicht als maschinenübergreifende SLA.


## Zitationen

Wenn du meine Software in deiner Forschung verwendest, zitiere mich bitte.

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
