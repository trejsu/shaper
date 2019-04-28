from .brush import Brush, QuadrangleBrush
from .curve import Curve
from .ellipse import Ellipse
from .quadrangle import Quadrangle, Rectangle
from .shape import Shape
from .triangle import Triangle


def from_index(index):
    return {
        0: Triangle,
        1: Rectangle,
        2: Ellipse,
        3: Quadrangle,
        4: QuadrangleBrush,
        5: Curve
    }[index]


def index_of(shape):
    return {
        Triangle: 0,
        Rectangle: 1,
        Ellipse: 2,
        Quadrangle: 3,
        QuadrangleBrush: 4,
        Curve: 5
    }[shape]
