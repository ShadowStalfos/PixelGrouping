from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossCECfg:
    weight: float


@dataclass
class LossCECfgWrapper:
    mse: LossCECfg


class LossMse(Loss[LossCECfg, LossCECfgWrapper]):
    def __init__(self, cfg: LossCECfgWrapper) -> None:
        super().__init__()

        self.loss = nn.CrossEntropyLoss()


    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt = batch["target"]["mask"]
        loss = self.loss(prediction.class_, gt)
        return self.cfg.weight * loss.mean() # is mean necessary here?
