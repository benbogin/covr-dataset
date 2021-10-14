import torch

from allennlp.training.learning_rate_schedulers import PolynomialDecay
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("linear_with_warmup_float")
class LinearWithWarmup(PolynomialDecay):
    """
    Same as `linear_with_warmup` except that it accepts a _proportional_ `warmup_steps`
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_steps_per_epoch: int,
        warmup_steps_prop: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            num_epochs,
            num_steps_per_epoch,
            power=1.0,
            warmup_steps=int(num_epochs*num_steps_per_epoch*warmup_steps_prop),
            end_learning_rate=0.0,
            last_epoch=last_epoch,
        )
