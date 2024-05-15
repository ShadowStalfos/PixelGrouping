from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor, nn, argmax
from torchvision.transforms.functional import rgb_to_grayscale as rgb2gray

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossCECfg:
    weight: float


@dataclass
class LossCECfgWrapper:
    crossentropy: LossCECfg


class LossCE(Loss[LossCECfg, LossCECfgWrapper]):
    def __init__(self, cfg: LossCECfgWrapper) -> None:
        super().__init__(cfg)

        self.loss = nn.CrossEntropyLoss()


    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt = batch["target"]["objects"].float().squeeze(0)
        x = argmax(prediction.class_, dim=2).float().permute(1, 0, 2, 3)
        loss = self.loss(x, gt)
        return self.cfg.weight * loss.mean()
