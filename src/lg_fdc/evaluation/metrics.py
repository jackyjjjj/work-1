from __future__ import annotations

from collections import Counter, defaultdict


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """普通分类准确率。"""

    _validate_lengths(y_true, y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """平衡准确率：先算每类召回率，再对类别平均。"""

    _validate_lengths(y_true, y_pred)
    labels = sorted(set(y_true))
    if not labels:
        return 0.0
    recalls = []
    for label in labels:
        total = sum(item == label for item in y_true)
        correct = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        recalls.append(correct / total if total else 0.0)
    return sum(recalls) / len(recalls)


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro-F1：先算每类 F1，再对类别平均。"""

    _validate_lengths(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for true, pred in zip(y_true, y_pred, strict=True):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    scores = []
    for label in labels:
        precision_den = tp[label] + fp[label]
        recall_den = tp[label] + fn[label]
        precision = tp[label] / precision_den if precision_den else 0.0
        recall = tp[label] / recall_den if recall_den else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def confusion_counts(y_true: list[str], y_pred: list[str]) -> dict[tuple[str, str], int]:
    """返回混淆矩阵计数，key 为 (真实标签, 预测标签)。"""

    _validate_lengths(y_true, y_pred)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for true, pred in zip(y_true, y_pred, strict=True):
        counts[(true, pred)] += 1
    return dict(counts)


def _validate_lengths(y_true: list[str], y_pred: list[str]) -> None:
    """评估指标的输入长度校验。"""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")