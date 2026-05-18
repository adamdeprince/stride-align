# stride-align

**Langues:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · **[Français](README.fr.md)** · [Español](README.es.md) · [Português do Brasil](README.pt-BR.md) · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` est une [bibliothèque fulgurante](BENCHMARK.md) qui te dit à quel point deux chaînes se ressemblent. Elle le fait en implémentant les algorithmes Smith-Waterman et Needleman-Wunsch. Plutôt que de faire un cours, on va apprendre en pratiquant. Entrons directement dans le fonctionnement.

## Installation

```bash
pip install stride-align
```

Sur les systèmes Loongson, installe NumPy depuis ta distribution Linux avant d’installer `stride-align` :

```bash
sudo apt install python3-numpy
pip install stride-align
```

D’abord, une précision : je n’utilise pas des textes religieux ici pour pousser une idée. Pour cette démo, j’ai besoin de plusieurs documents assez grands, dans le domaine public, qui ont le même sens mais des formulations différentes. La Bible se trouve simplement répondre à ce besoin de façon presque bizarrement parfaite.

Imaginons deux phrases ; utilisons la première phrase de la Genèse pour cela :

Dans l’American Standard Version, on a : "In the beginning God created the heavens and the earth."

Dans la King James Version, on a : "In the beginning God created the heaven and the earth."

À l’œil nu, on voit une différence : heavens contre heaven. Mais comment la quantifier ? On utilise ce petit morceau de code :

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Quand on l’exécute, il affiche :

```python
0.9907407407407407
```

Les scores normalisés sont compris entre `0` et `1`. Un score de `1` signifie que les entrées correspondent exactement avec le modèle de score par défaut. Des scores proches de `0` signifient que les entrées ont peu en commun, même si Smith-Waterman peut encore trouver de petits appariements locaux dans des chaînes par ailleurs sans rapport.

Changeons maintenant le texte et regardons ce qui arrive au score.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

et Python affiche :

```
0.12222222222222222
```

Tu vois l’idée ? Plus les chaînes se ressemblent, plus le score est élevé.

Construisons un exemple plus grand, pour sentir les performances de la bibliothèque. Tu remarqueras sans doute qu’on alterne entre Smith-Waterman et Needleman-Wunsch et tu peux te demander lequel utiliser quand. Utilise Needleman-Wunsch quand tu veux comparer toute l’entrée à toute l’entrée. Utilise Smith-Waterman quand tu veux trouver la meilleure région correspondante à l’intérieur d’entrées plus grandes.

Passons au code de démonstration. Il faut `requests` pour cette partie :

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

Comment utiliser cela ? Supposons que nous ayons un verset biblique au hasard et que nous voulions savoir de quel chapitre et de quel verset il vient. Tu dis `grep` ? Ah non, ciel, on a fait une erreur. Le verset vient d’une autre traduction, disons Catholic Public Domain, et ce que nous avons sur l’ordinateur est la King James Bible. La correspondance exacte de chaînes de `grep` ne fonctionnera pas ici. Comment trouver le chapitre et le verset ? On cherche bien sûr la chaîne « la plus proche » ou « la plus similaire » avec `stride-align`.

Dans notre démo, la première partie s’occupe du téléchargement et de la mise en cache. Les braves gens de [Open Bible](https://openbible.com) ont mis ce texte à portée de HTTP, mais nous voulons respecter leur budget informatique, donc nous mettons en cache ce que nous téléchargeons. C’est simplement être un bon citoyen du réseau.

Ensuite, nous chargeons toutes les lignes dans une liste. Nous supprimons les retours à la ligne et mettons tout en minuscules, parce que nous ne voulons pas chipoter sur la touche Maj.

Enfin, la boucle `while True:` récupère une ligne de texte, probablement le verset de la version Catholic de la Bible dont nous voulons retrouver le chapitre et le verset, puis la compare à toutes les lignes de la King James Bible avec la forme par lot de Needleman-Wunsch. Elle renvoie un tableau de scores. Nous utilisons `argmax()` pour trouver la ligne au meilleur score, puis nous affichons la ligne associée à cet indice. Essayons.

Je vais utiliser Jeremiah 4:28 de la Catholic Bible ; il est en fait assez différent du même verset dans la King James Bible. Voyons ce qui se passe…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

… et nous l’avons trouvé ! Et assez vite en plus.

## Autre démonstration : correction orthographique

C’est un correcteur orthographique jouet, pas un outil de production. Il ignore la ponctuation, les majuscules, la fréquence des mots, les noms propres et le contexte. Le but est de montrer le même motif « une requête contre beaucoup de candidats » sur une tâche familière.

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

La première chose que fait ce script est de chercher la liste de mots correctement orthographiés fournie par le système d’exploitation. Son emplacement peut varier d’une distribution à l’autre. Une fois trouvée, nous la chargeons, supprimons les retours à la ligne et lançons la correction.

La correction ressemble beaucoup à l’appariement précédent. Pour chaque mot candidat, nous le comparons à tous les mots de notre liste de mots corrects, utilisons `argmax()` pour trouver le candidat au meilleur score, puis remplaçons le mot par ce candidat. On pourrait accélérer cela avec des optimisations, par exemple en ne cherchant pas de correspondance pour les mots déjà corrects, mais c’est une démo et cette optimisation est laissée au lecteur.

Voyons ce que cela donne !

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Détails

L’échafaudage actuel fournit :

- alignement Needleman-Wunsch avec score seulement
- alignement Needleman-Wunsch avec rétrotraçage
- alignement Smith-Waterman avec score seulement
- alignement Smith-Waterman avec rétrotraçage
- une disposition de backends qui correspond au motif de spécialisation utilisé dans `massive-speedup`
- détection CPU/backend et répartition des backends côté Python

La frontière native accepte :

- `bytes` contre `bytes`
- `str` contre `str`
- des séquences d’objets Python immuables et hachables
- des entrées mixtes séquence/objet où le côté `str` ou `bytes` est traité comme une séquence

Les paires directes `bytes` contre `str` lèvent `TypeError`.

Les implémentations actuelles sont des noyaux génériques de programmation dynamique avec un prétraitement qui sérialise les entrées Python en flux de jetons de 8, 16, 32 ou 64 bits. Des backends spécialisés SIMD pourront remplacer plus tard les unités de traduction backend sans changer l’API Python.

Les fonctions à score seul renvoient des scores numériques. Les variantes normalisées renvoient des scores entre `0` et `1`. Les fonctions de chemin renvoient des objets de résultat d’alignement contenant le score, les valeurs alignées, les opérations et, quand c’est disponible, des résumés de style CIGAR.

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

Utilise `Scores(...).compare([...])` ou les fonctions `*_scores()` pour les charges de travail avec une requête contre beaucoup de cibles. Ce chemin prépare la requête/le profil une seule fois et constitue l’API de performance préférée pour les comparaisons répétées de textes anglais/chinois.

Les sorties de rétrotraçage préservent le type de chemin rapide apparié :

- `str` en entrée renvoie des `str` alignées
- `bytes` en entrée renvoie des `bytes` alignés
- les entrées séquence/objet renvoient des valeurs `tuple` alignées avec des trous `None`

Passe `width=8`, `16`, `32` ou `64` pour forcer la largeur interne des jetons/scores au lieu d’utiliser la sélection automatique.

Certaines fonctions exposent des chaînes CIGAR, abréviation de “Concise Idiosyncratic Gapped Alignment Report”. CIGAR est la notation compacte des opérations d’alignement utilisée par les outils SAM/BAM. Pour la version formelle complète, voir la [spécification SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Optimisations et bancs d’essai

Une attention sérieuse a été portée, et continue de l’être, aux performances de `stride-align`. La bibliothèque inclut des optimisations SIMD pour plusieurs cibles courantes, dont x86, Arm et LoongArch.

**LoongArch / Loongson.** L’histoire d’optimisation Loongson est particulièrement parlante : dans le cas de benchmark vérifié portant sur du texte anglais, une largeur de score de 16 bits et un Smith-Waterman à score seul, le backend LASX est 16 fois plus rapide que le backend générique et **22,4 fois** plus rapide que Parasail.

Si tu es chercheur, que tu utilises des serveurs Loongson et que tu profites de cette accélération, les citations, rapports de bogues, cas de banc d’essai et petits souvenirs chinois bon marché sont appréciés. Thé, marque-pages de calligraphie, ornements en papier découpé, porte-bonheur en nœud chinois, porte-clés panda et petits objets de bureau en forme de dragon sont les bienvenus. Merci de ne rien envoyer de cher ni rien qui exige des formalités douanières.

Voir les [bancs d’essai complets](BENCHMARKS.md).

## Micro-banc natif

Pour du profilage de performance sans trames Python ni orchestration de banc d’essai, configure une construction de micro-banc x86 native :

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` garde les modules nanobind non dépouillés et ajoute des symboles de débogage ainsi que des pointeurs de trame tout en conservant `-O3`.

La ligne de base du micro-banc natif enregistrée dans le dépôt se trouve dans `benchmarks/x86_microbench_baseline.json`. Traite-la comme un garde-fou local à seuil lâche, pas comme un SLA entre machines.


## Citations

Si tu utilises mon logiciel dans tes recherches, cite-moi s’il te plaît.

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
