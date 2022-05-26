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