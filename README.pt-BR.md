# stride-align

**Idiomas:** [English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Français](README.fr.md) · [Español](README.es.md) · **[Português do Brasil](README.pt-BR.md)** · [Русский](README.ru.md) · [Tiếng Việt](README.vi.md) · [Bahasa Indonesia](README.id.md) · [हिन्दी](README.hi.md) · [العربية](README.ar.md) · [Türkçe](README.tr.md) · [Polski](README.pl.md)

`stride-align` é uma [biblioteca absurdamente rápida](BENCHMARK.md) que diz quão “parecidas” duas strings são. Ela faz isso implementando os algoritmos Smith-Waterman e Needleman-Wunsch. Em vez de dar uma aula, vamos aprender fazendo. Vamos direto ao funcionamento.

## Instalação

```bash
pip install stride-align
```

Em sistemas Loongson, instale o NumPy pela sua distribuição Linux antes de instalar `stride-align`:

```bash
sudo apt install python3-numpy
pip install stride-align
```

Primeiro, um aviso: não estou usando textos religiosos aqui para empurrar uma agenda. Para esta demonstração, preciso de vários documentos razoavelmente grandes, de domínio público, que tenham o mesmo significado mas sejam redigidos de forma diferente. A Bíblia simplesmente atende a esse requisito de um jeito bizarramente perfeito.

Imagine que temos duas frases; vamos usar a primeira frase do Gênesis:

Na American Standard Version, temos: "In the beginning God created the heavens and the earth."

Na King James Version, temos: "In the beginning God created the heaven and the earth."

Dá para ver a diferença a olho nu: heavens contra heaven. Mas como quantificar essa diferença? Usamos este pequeno trecho de código:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

Quando executamos, ele imprime:

```python
0.9907407407407407
```

Pontuações normalizadas ficam entre `0` e `1`. Uma pontuação de `1` significa que as entradas são uma correspondência exata no modelo de pontuação padrão. Pontuações perto de `0` significam que as entradas têm pouco em comum, embora Smith-Waterman ainda possa encontrar pequenas correspondências locais dentro de strings que, no geral, não têm relação.

Agora vamos mudar o texto e ver o que acontece com a pontuação.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

e o Python imprime:

```
0.12222222222222222
```

Começando a entender? Quanto mais parecidas as strings, maior a pontuação.

Vamos montar um exemplo maior, algo que dê uma noção do desempenho da biblioteca. Você provavelmente vai notar que alternamos entre Smith-Waterman e Needleman-Wunsch e talvez se pergunte quando usar qual. Use Needleman-Wunsch quando quiser comparar a entrada inteira contra a entrada inteira. Use Smith-Waterman quando quiser encontrar a melhor região correspondente dentro de entradas maiores.

Certo, vamos ao código da demonstração. Você precisa de `requests` para esta parte:

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

Como podemos usar isso? Suponha que temos um versículo bíblico aleatório e queremos saber de que capítulo e versículo ele vem. `grep`, você diz? Ah, céus, não, cometemos um erro. O versículo vem de outra tradução, digamos Catholic Public Domain, e no computador temos a King James Bible. A correspondência exata de strings do `grep` não vai funcionar aqui. Como encontramos o capítulo e o versículo? Procuramos a string “mais próxima” ou “mais parecida” usando `stride-align`, é claro.

Na nossa demonstração, a primeira parte cuida de baixar e armazenar em cache. O pessoal gentil do [Open Bible](https://openbible.com) colocou esse texto em um local acessível por HTTP, mas queremos respeitar o orçamento de TI deles, então armazenamos em cache o que baixamos. É só boa cidadania.

Na parte seguinte carregamos todas as linhas em uma lista. Removemos quebras de linha e colocamos tudo em minúsculas porque não queremos ficar implicando com a tecla Shift.

Por fim, o laço `while True:` coleta uma linha de texto, presumivelmente o versículo da versão Catholic da Bíblia para o qual queremos descobrir capítulo e versículo, e compara essa linha com todas as linhas da King James Bible usando a forma em lote de Needleman-Wunsch. Ele retorna um array de pontuações. Usamos `argmax()` para encontrar a linha com maior pontuação e então imprimimos a linha associada a esse índice. Vamos tentar.

Vou usar Jeremiah 4:28 da Catholic Bible; ele é bem diferente do mesmo versículo na King James Bible. Vamos ver o que acontece…

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

… e encontramos! E bem rápido também.

## Outra demonstração: verificação ortográfica

Este é um corretor ortográfico de brinquedo, não um de produção. Ele ignora pontuação, capitalização, frequência de palavras, nomes próprios e contexto. O ponto é mostrar o mesmo padrão de uma consulta contra muitos candidatos em uma tarefa familiar.

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

A primeira coisa que este script faz é tentar encontrar a lista de palavras corretamente grafadas do sistema operacional. O local pode variar de distribuição para distribuição. Depois de encontrada, carregamos a lista, removemos as quebras de linha e começamos a verificação ortográfica.

A verificação se parece muito com a correspondência que fizemos antes. Para cada palavra candidata, comparamos contra todas as palavras da lista de palavras corretas, usamos `argmax()` para encontrar a candidata com maior pontuação e substituímos a palavra por ela. Poderíamos acelerar com otimizações, como não procurar correspondência para palavras já corretas, mas isto é uma demonstração e essa otimização fica como exercício para o leitor.

Vamos ver como funciona!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```


## Detalhes

A estrutura atual fornece:

- alinhamento Needleman-Wunsch somente com pontuação
- alinhamento Needleman-Wunsch com retrocesso
- alinhamento Smith-Waterman somente com pontuação
- alinhamento Smith-Waterman com retrocesso
- um layout de backends que combina com o padrão de especialização usado em `massive-speedup`
- detecção de CPU/backend e despacho de backend no lado Python

A fronteira nativa aceita:

- `bytes` contra `bytes`
- `str` contra `str`
- sequências de objetos Python imutáveis e hasháveis
- entradas mistas de sequência/objeto em que um lado `str` ou `bytes` é tratado como sequência

Pares diretos `bytes` contra `str` levantam `TypeError`.

As implementações atuais são núcleos genéricos de programação dinâmica com pré-processamento que serializa entradas Python em fluxos de tokens de 8, 16, 32 ou 64 bits. Backends especializados em SIMD podem substituir depois as unidades de tradução do backend sem mudar a API Python.

Funções somente de pontuação retornam pontuações numéricas. As variantes normalizadas retornam pontuações entre `0` e `1`. Funções de caminho retornam objetos de resultado de alinhamento contendo a pontuação, valores alinhados, operações e, quando disponível, resumos no estilo CIGAR.

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

Use `Scores(...).compare([...])` ou as funções `*_scores()` para cargas de trabalho de uma consulta contra muitos alvos. Esse caminho prepara a consulta/perfil uma vez e é a API de desempenho preferida para comparações repetidas de texto em inglês/chinês.

As saídas de retrocesso preservam o tipo de caminho rápido pareado:

- `str` de entrada retornam `str` alinhadas
- `bytes` de entrada retornam `bytes` alinhados
- entradas de sequência/objeto retornam valores `tuple` alinhados com lacunas `None`

Passe `width=8`, `16`, `32` ou `64` para forçar a largura interna de token/pontuação em vez de usar a seleção automática.

Algumas funções expõem strings CIGAR, abreviação de “Concise Idiosyncratic Gapped Alignment Report”. CIGAR é a notação compacta de operações de alinhamento usada por ferramentas SAM/BAM. Se quiser a versão formal completa, consulte a [especificação SAM](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Otimizações e benchmarks

A história de desempenho de `stride-align` recebeu, e continua recebendo, atenção cuidadosa. A biblioteca inclui otimização SIMD para vários alvos comuns, incluindo x86, Arm e LoongArch.

**LoongArch / Loongson.** A história de otimização do Loongson é especialmente reveladora: no caso de benchmark verificado com texto em inglês, largura de pontuação de 16 bits e Smith-Waterman somente de pontuação, o backend LASX é 16 vezes mais rápido que o backend genérico e **22,4 vezes** mais rápido que o Parasail.

Se você é pesquisador usando servidores Loongson e se beneficiando dessa aceleração, citações, relatórios de bugs, casos de benchmark e pequenos souvenirs chineses baratos são apreciados. Chá, marcadores de caligrafia, enfeites de papel recortado, pingentes de nó chinês, chaveiros de panda e pequenos objetos de mesa em forma de dragão são todos bem-vindos. Por favor, não envie nada caro nem nada que exija documentação alfandegária.

Veja os [benchmarks completos](BENCHMARKS.md).

## Microbenchmark nativo

Para perfilamento de desempenho sem quadros Python ou orquestração de benchmark, configure uma compilação nativa de microbenchmark x86:

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

`STRIDE_ALIGN_PERF_SYMBOLS=ON` mantém os módulos nanobind sem remoção de símbolos e adiciona símbolos de depuração mais ponteiros de quadro, preservando `-O3`.

A linha de base de microbenchmark nativo registrada fica em `benchmarks/x86_microbench_baseline.json`. Trate-a como uma proteção local com limite folgado, não como um SLA entre máquinas.


## Citações

Se você usar meu software na sua pesquisa, por favor cite-me.

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
