# 龙芯 3A6000（2.0 GHz 笔记本版）vs Tiger Lake i7（3.0 GHz，第 11 代）—— cdist 剪枝

**Languages:** [English](loongson-vs-tiger-lake-cdist-2026-05-24.md) · [简体中文](loongson-vs-tiger-lake-cdist-2026-05-24.zh-CN.md)

这是一份关于 2026-05-24 跨架构跑分的简短报告，跑的是
`tools/bench_cdist_pruning.py`。**笔记本版 2.0 GHz 龙芯 3A6000，
在六种归一化评分器中的五种里，在 T=0.99 这个阈值下，
超过了 3.0 GHz 的 Tiger Lake i7（英特尔第 11 代）**。而 T=0.99
正是模糊匹配 / 记录链接（record linkage）工作负载真正花时间的
高阈值剪枝场景。

两台机器都用 **GCC 15.2、`-O3`** 编译，同一份源码树，同一份基准
脚本。这已经是我能在本实验室里做到的、最接近“干净的架构对决”的
对比了。

龙芯核心微架构的参考资料：Chips and Cheese 对一颗 2.5 GHz 3A6000
样品的深度评测
(https://chipsandcheese.com/p/loongson-3a6000-a-star-among-chinese-cpus)。
我手上的 3A6000 是 **2.0 GHz** 的笔记本版，主频比 C&C 测的那颗
低 20%。

## 头条：剪枝阈值（T=0.99）

| 评分器 | Tiger Lake i7（3.0 GHz） | 龙芯 3A6000（2.0 GHz） | 龙芯 / Tiger Lake | 龙芯 / Tiger Lake（主频归一） |
| --- | ---: | ---: | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        | 290M | 370M | **1.28x** | **1.91x** |
| DAMERAU_LEVENSHTEIN_NORMALIZED| 297M | 426M | **1.43x** | **2.15x** |
| HAMMING_NORMALIZED (n=100)    | 136M | 139M | 1.02x | 1.54x |
| INDEL_NORMALIZED              | 286M | 402M | **1.41x** | **2.11x** |
| JARO                          | 129M | 190M | **1.47x** | **2.21x** |
| JARO_WINKLER                  | 141M | 136M | 0.96x | 1.45x |

按主频归一后（pairs/sec/GHz），龙芯在每一种评分器上都比
Tiger Lake **快 1.45 倍到 2.21 倍**。3A6000 不只是用低 33% 的
主频跑出了更高的绝对吞吐，而是在每个时钟周期里完成了大约两倍的
工作。

## 反转：原始 SIMD 内核（T=0，无剪枝）

同样的评分器，不设阈值，每一对都要走完整的 SIMD：

| 评分器 | Tiger Lake（T=0） | 龙芯（T=0） | TL / LS | TL / LS（主频归一） |
| --- | ---: | ---: | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        | 0.49M | 0.31M | 1.58x | 1.05x |
| DAMERAU_LEVENSHTEIN_NORMALIZED| 0.50M | 0.32M | 1.56x | 1.04x |
| HAMMING_NORMALIZED            | 0.46M | 0.29M | 1.59x | 1.06x |
| INDEL_NORMALIZED              | 0.47M | 0.29M | 1.62x | 1.08x |
| JARO                          | 0.53M | 0.30M | 1.77x | 1.18x |
| JARO_WINKLER                  | 0.51M | 0.29M | 1.76x | 1.17x |

从绝对吞吐看，Tiger Lake 的更宽 SIMD（AVX-512 = 512 位 = 每个向量
8 个 u64 lane，LASX = 256 位 = 4 个 u64 lane）赢大约 60%。
**但是主频归一以后，两条 SIMD 路径在 Lev / OSA / Hamming / Indel
上基本打平，龙芯在 Jaro / JW 上还反超 15% 到 18%。** 也就是说，
Tiger Lake 的绝对优势几乎完全来自于它的主频高 1.5 倍，而不是来自
位并行吞吐本身更强。

## 那剪枝一介入，龙芯为什么反过来赢了？

在 T=0.99、长度 4～40 的随机字符串上，长度差剪枝会在任何 SIMD
工作开始之前，就把大约 600 对里的 599 对剔掉。剩下的实际耗时
基本上都是**标量簿记开销**：

1. 算每对的 `max_normalized_similarity` 上界（几次 `std::min` /
   `std::max`，再加一次除法）。
2. 判断这个上界够不够阈值。
3. 为幸存下来的对构造候选子列表。
4. 计算每对的距离 cutoff。
5. *然后* 才在幸存者上跑 SIMD。

第 1～4 步全是标量整数 / 浮点工作，跟 SIMD 没关系。在这种混合负载
上，3A6000 **每个时钟周期的吞吐大约是 Tiger Lake 核心的两倍**。

把同一台机器上 T=0.99 / T=0 的加速比拿出来对比，这个每时钟优势
看得最清楚：

| 评分器 | Tiger Lake | 龙芯 |
| --- | ---: | ---: |
| LEVENSHTEIN_NORMALIZED        |   587x | **1,199x** |
| DAMERAU_LEVENSHTEIN_NORMALIZED|   598x | **1,353x** |
| HAMMING_NORMALIZED            |   293x |   487x |
| INDEL_NORMALIZED              |   611x | **1,408x** |
| JARO                          |   245x |   642x |
| JARO_WINKLER                  |   275x |   472x |

龙芯把剪枝转化为实际耗时节省的效率，是 Tiger Lake 的 2 到
2.3 倍。

## 这意味着什么

这组结果分得很清楚：

* **整矩阵 `cdist` 类工作负载**，也就是只写 `scorer=...`、不过阈值
  也不取 top-k 的场景，绝对吞吐上 Tiger Lake 领先约 60%。AVX-512
  的更宽向量在这里占主导。
* **`cdist_above_threshold` / `cdist_top_k` 在有意义阈值下的工作负载**，
  也就是绝大多数生产环境里模糊匹配代码真正跑的负载，龙芯领先
  28% 到 47%。这一档的热路径是每对的标量簿记，而 3A6000 在那个
  路径上的每时钟性能实打实更强。

两个结论都有意义。第一个是各种规格表暗示的“SIMD 吞吐”层面的对比。
第二个是用户在生产里真正会体验到的对比：在同一份源码、同一份
编译器下，一颗基于 LoongArch 架构的 2.0 GHz 龙芯 3A6000 笔记本芯片
明显胜过了 3.0 GHz 的 Tiger Lake i7。

## 复现方式

```sh
# Tiger Lake 主机
tools/bench_cdist_pruning.py --scorer LEVENSHTEIN_NORMALIZED
tools/bench_cdist_pruning.py --scorer DAMERAU_LEVENSHTEIN_NORMALIZED
tools/bench_cdist_pruning.py --scorer INDEL_NORMALIZED
tools/bench_cdist_pruning.py --scorer JARO
tools/bench_cdist_pruning.py --scorer JARO_WINKLER
tools/bench_cdist_pruning.py --scorer HAMMING_NORMALIZED --min-len 100 --max-len 100

# 龙芯主机 —— 同一份脚本，同一份源码
LD_LIBRARY_PATH=/opt/loongson-gcc-15.2.0/lib \
    .venv/bin/python tools/bench_cdist_pruning.py --scorer LEVENSHTEIN_NORMALIZED
# 其余同理
