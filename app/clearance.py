"""轮廓间安全间隙校验。

基于 :func:`app.svggeo.normalize_path` 的层级分类结果，对参与校验的
轮廓按**细分后的折线**计算两两边界的最小欧氏距离，报告距离小于
阈值的冲突对。

规则：

- 参与对象：闭合非退化轮廓（``outer`` / ``hole``）；
  ``include_open=True`` 时额外纳入开放轮廓（按未闭合折线计）；
  退化轮廓（``degenerate``）始终跳过；同一轮廓不与自身比较；
- 距离定义：两轮廓细分线段集合之间的最小线段-线段欧氏距离，
  边界相交或相接时距离为 0；
- **距离等于阈值视为通过**：距离先量化到 ``CLEARANCE_DECIMALS`` 位
  小数再与阈值严格比较（``<`` 才算冲突），吸收浮点噪声；
- 纯计算：不修改方向、不写数据库，相同输入必然得到相同输出。
"""

from __future__ import annotations

import math
from typing import Any, Iterator

from .svggeo import dedupe_ring, flatten_subpath, normalize_path, parse_path

CLEARANCE_DECIMALS = 9  # 距离量化位数：等于阈值视为通过的浮点容差

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# 线段几何
# ---------------------------------------------------------------------------

def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """点到线段的最短距离（零长度线段退化为点到点）。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _cross(a: Point, b: Point, c: Point) -> float:
    """向量 ab 与 ac 的叉积（符号表示 c 在 ab 的哪一侧）。"""
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    """p 是否落在线段 ab 的包围盒内（调用前已知共线）。"""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """两线段是否相交（含端点相接与共线重叠）。"""
    o1 = _cross(a, b, c)
    o2 = _cross(a, b, d)
    o3 = _cross(c, d, a)
    o4 = _cross(c, d, b)
    # 一般情形：严格跨立
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) \
            and o1 != 0.0 and o2 != 0.0 and o3 != 0.0 and o4 != 0.0:
        return True
    # 退化情形：共线且端点落在对方线段上
    if o1 == 0.0 and _on_segment(a, b, c):
        return True
    if o2 == 0.0 and _on_segment(a, b, d):
        return True
    if o3 == 0.0 and _on_segment(c, d, a):
        return True
    if o4 == 0.0 and _on_segment(c, d, b):
        return True
    return False


def segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    """两线段的最小欧氏距离；相交（含相接）时为 0。"""
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


# ---------------------------------------------------------------------------
# 轮廓边界距离
# ---------------------------------------------------------------------------

def _segments(points: list[Point], closed: bool
              ) -> Iterator[tuple[Point, Point]]:
    """折线的线段序列；闭合轮廓追加末点到首点的闭合边。"""
    for k in range(len(points) - 1):
        yield points[k], points[k + 1]
    if closed and len(points) > 1:
        yield points[-1], points[0]


def _seg_bbox(p: Point, q: Point) -> tuple[float, float, float, float]:
    return (min(p[0], q[0]), min(p[1], q[1]),
            max(p[0], q[0]), max(p[1], q[1]))


def _bbox_gap(box_a: tuple[float, float, float, float],
              box_b: tuple[float, float, float, float]) -> float:
    """两个轴对齐包围盒之间的最小距离（相交时为 0）。"""
    dx = max(box_a[0] - box_b[2], box_b[0] - box_a[2], 0.0)
    dy = max(box_a[1] - box_b[3], box_b[1] - box_a[3], 0.0)
    return math.hypot(dx, dy)


def boundary_distance(points_a: list[Point], closed_a: bool,
                      points_b: list[Point], closed_b: bool) -> float:
    """两条细分折线边界的最小距离（闭合轮廓含闭合边）。

    用线段包围盒间隙作为下界剪枝：下界已不小于当前最优值时跳过
    精确计算，保证 O(n·m) 最坏复杂度下的常见情形接近线性。
    """
    segs_b = [(_seg_bbox(c, d), c, d)
              for c, d in _segments(points_b, closed_b)]
    best = math.inf
    for a, b in _segments(points_a, closed_a):
        box_a = _seg_bbox(a, b)
        for box_b, c, d in segs_b:
            if _bbox_gap(box_a, box_b) >= best:
                continue
            dist = segment_distance(a, b, c, d)
            if dist < best:
                best = dist
                if best == 0.0:
                    return 0.0
    return best


# ---------------------------------------------------------------------------
# 校验主流程
# ---------------------------------------------------------------------------

def check_clearance(d: str, fill_rule: str, min_distance: float,
                    include_open: bool = False) -> dict[str, Any]:
    """对一条 SVG path 做轮廓间安全间隙校验（即时规范化，不写库）。

    参数:
        d: 原始 SVG path 数据
        fill_rule: ``nonzero`` 或 ``evenodd``
        min_distance: 最小安全间隙阈值，必须为正数
        include_open: 是否将开放轮廓纳入校验（退化轮廓始终跳过）

    返回:
        ``{min_distance, include_open, conflicts, summary}``；
        冲突对按 ``(a_original_index, b_original_index)`` 升序。
    """
    if not min_distance > 0 or math.isnan(min_distance):
        raise ValueError("min_distance 必须为正数")

    norm = normalize_path(d, fill_rule)
    subs = parse_path(d)
    # 与 normalize_path 内部一致的细分折线（按原始索引对齐）
    points = [dedupe_ring(flatten_subpath(s)) for s in subs]

    eligible: list[dict[str, Any]] = []
    for c in norm["contours"]:
        role = c["role"]
        if role not in ("outer", "hole") and not (include_open and role == "open"):
            continue
        idx = c["original_index"]
        if len(points[idx]) < 2:
            continue  # 防御：单点轮廓无边界线段
        eligible.append({
            "index": idx,
            "role": role,
            "closed": c["closed"],
            "parent": c["parent_original_index"],
            "points": points[idx],
        })

    conflicts: list[dict[str, Any]] = []
    pairs_checked = 0
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]
            pairs_checked += 1
            dist = boundary_distance(a["points"], a["closed"],
                                     b["points"], b["closed"])
            dist = round(dist, CLEARANCE_DECIMALS)
            if dist < min_distance:  # 等于阈值视为通过
                conflicts.append({
                    "a_original_index": a["index"],
                    "a_role": a["role"],
                    "b_original_index": b["index"],
                    "b_role": b["role"],
                    "distance": dist,
                    "threshold": min_distance,
                    # 孔洞与其直接父轮廓的间隙（壁厚）单独标记
                    "hole_parent_pair": (
                        (a["role"] == "hole" and a["parent"] == b["index"])
                        or (b["role"] == "hole" and b["parent"] == a["index"])),
                })

    # eligible 本身按原始索引升序，此处显式排序以保证确定性契约
    conflicts.sort(key=lambda e: (e["a_original_index"], e["b_original_index"]))

    return {
        "min_distance": min_distance,
        "include_open": include_open,
        "conflicts": conflicts,
        "summary": {
            "contours_total": len(norm["contours"]),
            "contours_checked": len(eligible),
            "pairs_checked": pairs_checked,
            "conflict_count": len(conflicts),
        },
    }
