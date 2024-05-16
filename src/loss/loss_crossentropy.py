from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor, nn, log, tensor
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

        self.loss = nn.CrossEntropyLoss(reduction="none")


    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt = batch["target"]["objects"].squeeze()
        x = prediction.class_.float().squeeze(0)
        loss = self.loss(x, gt)
        loss = loss / log(tensor(200))
        return self.cfg.weight * loss.mean()
