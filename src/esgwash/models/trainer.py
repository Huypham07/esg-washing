"""Vong train chung (spec 02 #3): seed control, pos_weight, eval, luu artifact.

Moi ket qua chinh chay 5 seeds -> mean +- std. Luu config snapshot canh model.
"""


def train_run(model_cls, config: dict, seed: int) -> dict:
    raise NotImplementedError  # TODO(Phase B1)


def multi_seed(model_cls, config: dict) -> dict:
    """Chay het config['train']['seeds'], gop metrics mean/std."""
    raise NotImplementedError
