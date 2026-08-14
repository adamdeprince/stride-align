# 编程接口（API）

stride-align 通过 Python、R、DuckDB、PostgreSQL 与 Memgraph 公开其原生算法。
本页为每种算法都给出五个接口的调用方式。任意示例上的语言选择器都会与
本页其他选择器及首页同步，因此无论读到哪里都可以切换接口。

以下示例假定 Python 中已经执行 `import stride_align as sa`，R 中已经执行
`library(stridealign)`；数据库面板则使用各自代码中显示的 Python 客户端。
请参阅[中文首页的安装演练](../../index.zh-CN.html#install)，按版本、平台与
处理器选择并[下载 DuckDB 扩展](https://distribution.goblinreactor.com/stride-align/duckdb/index.zh-CN.html)，
或[下载 Memgraph 查询模块](https://distribution.goblinreactor.com/stride-align/memgraph/index.zh-CN.html)。
PostgreSQL 16–18 使用 [PGXS 构建指南（英文）](../../bindings/postgres/)。

本编程接口按算法而不是语言适配器组织。Python 与 R 使用普通函数调用；
DuckDB 与 PostgreSQL 提供 `stride_*` SQL 函数；Memgraph 提供
`stride_align.*` Cypher 函数和流式 `cdist` 过程。数据库示例中的 Python
只负责连接、装载数据并提交原生 SQL 或 Cypher 操作。本原生跨语言参考不包含
兼容层。

## 向量化语义

以 `Q` 表示查询数量，`T` 表示目标数量，`K` 表示请求保留的结果数量。
函数名称同时决定 `Q × T` 比较网格中保留多少结果，以及调用方收到的数据
形状。

| 操作 | 逻辑输入 | 逻辑输出 |
| --- | --- | --- |
| `*_scores` / `*_similarities` | 1 个查询与 `T` 个目标 | 按目标顺序排列的 `T` 个分数 |
| `*_best` | 1 个查询与 `T` 个目标 | 1 个匹配；候选集为空时没有匹配 |
| `*_top_k` / `extract` | 1 个查询与 `T` 个目标 | 这些目标中最多 `K` 个匹配 |
| `cdist` | `Q` 个查询与 `T` 个目标 | 稠密的 `Q × T` 分数矩阵；Memgraph 则流式返回 `Q·T` 条配对记录 |
| `cdist_above_threshold` | 概念上的 `Q × T` 网格 | 仅返回达到阈值的配对记录 |
| `cdist_top_k` | 概念上的 `Q × T` 网格 | **整个网格全局**最多 `K` 条配对记录 |
| `cdist_top_k_per_query` | 概念上的 `Q × T` 网格 | `Q` 个分组，每个查询各自最多 `K` 个目标 |

五个宿主使用各自的原生数据类型表达这些逻辑形状：

| 宿主 | 集合与索引约定 |
| --- | --- |
| Python | 分数向量和稠密 `cdist` 结果是 NumPy 数组。排名使用元组与列表；阈值形式和逐查询形式使用迭代器。匹配索引从零开始。 |
| R | 结果使用数值向量、矩阵、数据框与列表。匹配索引从一开始。成对评分函数也能处理等长字符向量，或者将长度为一的一侧广播。 |
| DuckDB | 结果使用 `LIST`、嵌套 `LIST` 与 `STRUCT` 值。匹配记录中的索引字段从零开始，而 SQL 列表下标从一开始。应用于表列的成对函数通过 DuckDB 的向量化数据块机制运行；列表函数还可以在每一行内批量处理候选集合。 |
| PostgreSQL | 结果使用原生数组与 `jsonb` 匹配记录。SQL 数组下标从一开始；匹配记录中的来源索引仍从零开始。原生批量函数直接接收 `text[]` 集合。 |
| Memgraph | 标量函数在 Cypher 记录流上组合使用，不注册复数形式、提取或前 k 项函数。`cdist` 是批量读取过程，按行优先顺序返回从零开始的 `query_index`、`target_index` 与 `score`，调用方无需构造矩阵。 |

对于不使用替换矩阵的字符串评分器，阈值和全配对前 k 项筛选使用归一化
相似度评分器；稠密 `cdist` 也支持原始距离。匹配分数、不匹配分数、空位、
前缀、截断值与替换矩阵等选项，会传递给支持相应评分器的集合形式。

> `cdist_top_k(..., k=3)` 即使面对数千个查询，整个网格也最多只返回三项。
> `cdist_top_k_per_query(..., k=3)` 则让每个查询独立竞争，因此最多可返回
> `3 × Q` 项。需要全局记录关联时使用前者；需要为每个查询寻找近邻时使用
> 后者。

在 Memgraph 中，这些筛选写在 `CALL stride_align.cdist` 之后：`WHERE` 应用阈值，
`ORDER BY ... LIMIT` 选出全局前 k 项，分组后的 `collect(...)[..k]` 则为每个
查询保留各自的前 k 项。筛选留在 Cypher 中执行，而过程仍然逐步产生配对记录。

<!-- stride-vectorization-guide -->

## 算法

<!-- stride-api-catalog -->

每种算法下方都链接了参数、返回值、替换矩阵与回溯语义的详细英文参考。
