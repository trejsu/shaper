import numpy as np
import pytest

from shaper.shape.quadrangle import Quadrangle


def test_from_normalized_params():
    normalized = [1, 0.2866666666666667, 0.85, 0.32666666666666666, 0.65, 0.013333333333333334, 0.8,
                  -0.02, 0.5]
    w = 100
    h = 150
    expected_points = np.array([[100, 43], [85, 49], [65, 2], [80, -3]])
    r = Quadrangle.from_normalized_params(w, h, *normalized)
    assert np.array_equal(r.points, expected_points)
    assert r.alpha == 0.5


def test_normalized_params():
    r = Quadrangle.from_params(*[100, 43, 85, 49, 65, 2, 80, -3, 0.5])
    w = 100
    h = 150
    expected = np.array(
        [1, 0.2866666666666667, 0.85, 0.32666666666666666, 0.65, 0.013333333333333334, 0.8,
         -0.02, 0.5])
    normalized = r.normalized_params(w, h)
    assert np.array_equal(normalized, expected)


@pytest.mark.parametrize(
    "edge_start, edge_end, start_new_x_sign, start_new_y_sign, end_new_x_sign, end_new_y_sign", [
        (0, 1, 1, 0, 1, 0),
        (1, 2, 0, 1, 1, 0),
        (2, 3, -1, 0, -1, 0),
        (3, 0, -1, 0, 0, -1)
    ])
def test_without_edge(edge_start, edge_end, start_new_x_sign, start_new_y_sign, end_new_x_sign,
    end_new_y_sign):
    points = np.array([[-51, 159], [-16, 82], [37, 60], [125, 114]])
    q = Quadrangle.without_edge(points=points.copy(), edge=[edge_start, edge_end], alpha=1)
    assert np.sign(q.points[edge_start][0] - points[edge_start][0]) == start_new_x_sign
    assert np.sign(q.points[edge_start][1] - points[edge_start][1]) == start_new_y_sign
    assert np.sign(q.points[edge_end][0] - points[edge_end][0]) == end_new_x_sign
    assert np.sign(q.points[edge_end][1] - points[edge_end][1]) == end_new_y_sign
