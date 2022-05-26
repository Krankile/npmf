from utils.training import EarlyStop


def test_reset_stopper():
    stopper = EarlyStop(patience=2, min_delta=0.0)
    stopper([0.5, 0.5])
    stopper([1, 1])

    assert stopper.best_loss == 0.5
    assert stopper.triggers == 1

    stopper.reset()

    assert stopper.best_loss == float("inf")
    assert stopper.triggers == 0


def test_stopper_stops():
    stopper = EarlyStop(patience=2, min_delta=0.0)
    stopper([0.5, 0.5])
    assert not stopper([1, 1])
    assert stopper([1, 1])


def test_stopper_min_delta():
    stopper = EarlyStop(patience=2, min_delta=0.1)
    stopper([0.5, 0.5])
    stopper([0.5, 0.6])

    assert stopper.best_loss == 0.5
    assert stopper.triggers == 1

    stopper([0.3, 0.4])

    assert stopper.best_loss == 0.35
    assert stopper.triggers == 0
