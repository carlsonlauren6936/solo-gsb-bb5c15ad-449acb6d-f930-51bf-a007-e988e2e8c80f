"""clearance 间隙校验的几何与主流程单元测试。"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.clearance import (  # noqa: E402
    boundary_distance,
    check_clearance,
    point_segment_distance,
    segment_distance,
    segments_intersect,
)


# ---------------------------------------------------------------------------
# 线段几何
# ---------------------------------------------------------------------------

def test_point_segment_distance_foot_inside_and_outside():
    assert point_segment_distance((5, 3), (0, 0), (10, 0)) == 3.0   # 垂足在线段内
    assert point_segment_distance((-4, 3), (0, 0), (10, 0)) == 5.0  # 垂足在线段外 -> 端点
    assert point_segment_distance((0, 0), (0, 0), (10, 0)) == 0.0   # 端点上
    # 零长度线段退化为点到点
    assert point_segment_distance((3, 4), (0, 0), (0, 0)) == 5.0


def test_segments_intersect_crossing_touching_collinear():
    assert segments_intersect((0, 0), (10, 0), (5, -5), (5, 5))    # 十字交叉
    assert segments_intersect((0, 0), (10, 0), (10, 0), (10, 5))   # 端点相接
    assert segments_intersect((0, 0), (10, 0), (5, 0), (15, 0))    # 共线重叠
    assert segments_intersect((0, 0), (10, 0), (10, 0), (15, 0))   # 共线端点相接
    assert not segments_intersect((0, 0), (10, 0), (0, 1), (10, 1))    # 平行分离
    assert not segments_intersect((0, 0), (10, 0), (11, 0), (15, 0))   # 共线分离
    assert not segments_intersect((0, 0), (1, 0), (2, 1), (3, -1))     # 交叉点在外侧


def test_segment_distance_parallel_crossing_skew():
    assert segment_distance((0, 0), (10, 0), (0, 3), (10, 3)) == 3.0
    assert segment_distance((0, 0), (10, 0), (5, -1), (5, 1)) == 0.0
    # 错开线段：最近距离为端点 (1,0) 到 (3,4)
    d = segment_distance((0, 0), (1, 0), (3, 4), (4, 4))
    assert math.isclose(d, math.hypot(2, 4), abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 轮廓边界距离
# ---------------------------------------------------------------------------

def test_boundary_distance_nested_squares():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inner = [(2, 2), (4, 2), (4, 4), (2, 4)]
    assert boundary_distance(outer, True, inner, True) == 2.0
    # 参数对称
    assert boundary_distance(inner, True, outer, True) == 2.0


def test_boundary_distance_closed_flag_adds_closing_edge():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    probe = [(-2, 5), (-2, 5)]  # 左侧的点探针（零长度线段）
    # 闭合：左边 x=0 参与 -> 距离 2
    assert boundary_distance(square, True, probe, False) == 2.0
    # 开放：缺少闭合边，最近点为 (0,0)/(0,10)
    d = boundary_distance(square, False, probe, False)
    assert math.isclose(d, math.hypot(2, 5), abs_tol=1e-12)


def test_boundary_distance_intersecting_is_zero():
    a = [(0, 0), (20, 0), (20, 20), (0, 20)]
    b = [(10, -5), (30, -5), (30, 15), (10, 15)]
    assert boundary_distance(a, True, b, True) == 0.0


# ---------------------------------------------------------------------------
# check_clearance 主流程
# ---------------------------------------------------------------------------

def test_hole_parent_gap_flagged():
    d = "M0,0h40v40h-40Z M10,10h20v20h-20Z"  # 壁厚 10
    res = check_clearance(d, "nonzero", 15.0)
    assert len(res["conflicts"]) == 1
    c = res["conflicts"][0]
    assert c["a_original_index"] == 0 and c["a_role"] == "outer"
    assert c["b_original_index"] == 1 and c["b_role"] == "hole"
    assert c["distance"] == 10.0
    assert c["threshold"] == 15.0
    assert c["hole_parent_pair"] is True
    assert res["summary"] == {"contours_total": 2, "contours_checked": 2,
                              "pairs_checked": 1, "conflict_count": 1}


def test_equal_to_threshold_passes():
    d = "M0,0h40v40h-40Z M10,10h20v20h-20Z"
    assert check_clearance(d, "nonzero", 10.0)["conflicts"] == []
    assert check_clearance(d, "nonzero", 10.001)["conflicts"] != []


def test_sibling_holes_not_marked_as_parent_pair():
    d = ("M0,0h100v100h-100Z "
         "M10,10h20v20h-20Z "   # 孔1: x∈[10,30]
         "M35,10h20v20h-20Z")   # 孔2: x∈[35,55]，孔间距 5
    res = check_clearance(d, "nonzero", 6.0)
    # 孔1/孔2 距外环均为 10 -> 不冲突；孔1-孔2 距离 5 -> 冲突
    assert len(res["conflicts"]) == 1
    c = res["conflicts"][0]
    assert (c["a_original_index"], c["b_original_index"]) == (1, 2)
    assert c["a_role"] == c["b_role"] == "hole"
    assert c["distance"] == 5.0
    assert c["hole_parent_pair"] is False
    assert res["summary"]["pairs_checked"] == 3


def test_open_contour_excluded_by_default_included_on_demand():
    d = "M0,0h40v40h-40Z M5,5 L35,5 L35,35"  # 开放折线距外环 5
    res = check_clearance(d, "nonzero", 10.0)
    assert res["conflicts"] == []
    assert res["summary"]["contours_checked"] == 1

    res = check_clearance(d, "nonzero", 10.0, include_open=True)
    assert len(res["conflicts"]) == 1
    c = res["conflicts"][0]
    assert c["a_role"] == "outer" and c["b_role"] == "open"
    assert c["distance"] == 5.0
    assert c["hole_parent_pair"] is False


def test_degenerate_contour_always_skipped():
    d = "M0,0h40v40h-40Z M5,5 L30,5 L5,5 Z"  # 第二个轮廓面积 0 -> 退化
    res = check_clearance(d, "nonzero", 100.0, include_open=True)
    assert res["conflicts"] == []
    assert res["summary"]["contours_checked"] == 1


def test_conflicts_sorted_and_result_deterministic():
    d = ("M0,0h10v10h-10Z "      # 0
         "M12,0h10v10h-10Z "     # 1：距 0 为 2
         "M24,0h10v10h-10Z")     # 2：距 1 为 2，距 0 为 14
    res = check_clearance(d, "nonzero", 3.0)
    pairs = [(c["a_original_index"], c["b_original_index"])
             for c in res["conflicts"]]
    assert pairs == [(0, 1), (1, 2)]
    assert all(c["distance"] == 2.0 for c in res["conflicts"])
    assert all(c["hole_parent_pair"] is False for c in res["conflicts"])
    # 确定性：相同输入两次计算完全一致
    assert check_clearance(d, "nonzero", 3.0) == res


def test_intersecting_outers_zero_distance():
    d = "M0,0h20v20h-20Z M10,-5h20v20h-20Z"  # 两个相交的外环
    res = check_clearance(d, "nonzero", 1.0)
    assert len(res["conflicts"]) == 1
    assert res["conflicts"][0]["distance"] == 0.0


def test_curved_contour_uses_flattened_segments():
    # 圆形孔（两段弧）的最左极点 (15,20) 距外环左边 x=0 恰好 15
    d = ("M0,0h40v40h-40Z "
         "M15,20 A5,5 0 1 1 25,20 A5,5 0 1 1 15,20 Z")
    res = check_clearance(d, "nonzero", 16.0)
    assert len(res["conflicts"]) == 1
    c = res["conflicts"][0]
    assert c["b_role"] == "hole" and c["hole_parent_pair"] is True
    assert c["distance"] == 15.0
    # 曲线轮廓上等于阈值同样视为通过
    assert check_clearance(d, "nonzero", 15.0)["conflicts"] == []


def test_single_contour_no_pairs():
    res = check_clearance("M0,0h10v10h-10Z", "nonzero", 100.0)
    assert res["conflicts"] == []
    assert res["summary"]["pairs_checked"] == 0
    assert res["summary"]["contours_checked"] == 1


def test_invalid_min_distance_rejected():
    import pytest
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            check_clearance("M0,0h10v10h-10Z", "nonzero", bad)
