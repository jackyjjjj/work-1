from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from lg_fdc.data.manifest import ImageRecord


@dataclass(frozen=True)
class EpisodeConfig:
    """N-way K-shot episode 的采样配置。"""

    n_way: int
    k_shot: int
    q_queries: int
    seed: int = 0


@dataclass(frozen=True)
class Episode:
    """一次 few-shot 任务。

    support 用来构建类别原型，query 用来评估分类效果。
    """

    classes: tuple[str, ...]
    support: tuple[ImageRecord, ...]
    query: tuple[ImageRecord, ...]


def sample_episode(records: list[ImageRecord], config: EpisodeConfig) -> Episode:
    """从样本列表中采样一个 N-way K-shot episode。"""

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    # 每个被选中的类别至少要有 k_shot 个 support 和 q_queries 个 query。
    min_items = config.k_shot + config.q_queries
    eligible = sorted(label for label, items in grouped.items() if len(items) >= min_items)
    if len(eligible) < config.n_way:
        raise ValueError(
            f"Need at least {config.n_way} labels with {min_items} items each; "
            f"found {len(eligible)} eligible labels."
        )

    rng = random.Random(config.seed)
    classes = tuple(rng.sample(eligible, config.n_way))
    support: list[ImageRecord] = []
    query: list[ImageRecord] = []
    for label in classes:
        items = list(grouped[label])
        rng.shuffle(items)
        support.extend(items[: config.k_shot])
        query.extend(items[config.k_shot : config.k_shot + config.q_queries])

    # 打乱顺序，避免模型或评估代码隐式依赖类别排列。
    rng.shuffle(support)
    rng.shuffle(query)
    return Episode(classes=classes, support=tuple(support), query=tuple(query))