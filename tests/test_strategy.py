import numpy as np
import pytest

from es.strategy import RandomStrategy
from shaper.shape.shape import Shape


@pytest.mark.parametrize("strategy_class", [
    RandomStrategy
])
def test_ask_should_return_array_of_len_num_shapes(strategy_class):
    num_shapes = 10
    strategy = strategy_class(
        n=10,
        w=10,
        h=10,
        alpha=1,
        rng=np.random.RandomState(seed=9),
        decay=0.0003,
        shape_mode=0
    )
    shapes = strategy.ask(action=1)
    assert len(shapes) == num_shapes


@pytest.mark.parametrize("strategy_class", [
    RandomStrategy
])
def test_ask_should_return_array_of_shape_objects(strategy_class):
    strategy = strategy_class(
        n=10,
        w=10,
        h=10,
        alpha=1,
        rng=np.random.RandomState(seed=9),
        decay=0.0003,
        shape_mode=0
    )
    shapes = strategy.ask(action=1)
    assert all([isinstance(shape, Shape) for shape in shapes])


@pytest.mark.parametrize("strategy_class", [
    RandomStrategy
])
def test_result_should_return_the_best_score(strategy_class):
    num_shapes = 100
    strategy = strategy_class(
        n=num_shapes,
        w=10,
        h=10,
        alpha=1,
        rng=np.random.RandomState(seed=9),
        decay=0.0003,
        shape_mode=0
    )
    _ = strategy.ask(action=1)
    scores = np.random.randint(100, size=(num_shapes,))
    strategy.tell(scores)
    _, score = strategy.result()
    assert score == scores[np.argmin(scores)]
