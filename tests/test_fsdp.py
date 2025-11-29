from training.fsdp_minimal import train_one_step


def test_train_one_step_has_finite_loss() -> None:
    loss = train_one_step()
    assert isinstance(loss, float)
    assert loss > 0.0
    assert loss < 100.0