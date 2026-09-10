"""
Modelos baseline para la comparacion de la Tabla 8 del paper.

    - ResNet50      : torchvision ResNet-50 (stem conv7x7 + maxpool), 1 canal de
                      entrada, salida multi-label Sigmoid.
    - ResNet50-A    : ResNet50 + bloque SE (atencion de canal) despues de cada
                      una de las 4 etapas.
    - TResNet-M     : TResNet-M = misma familia que TResNet-L pero con
                      repeticiones (3, 4, 11, 3), sin SE.
    - TResNet-MA    : TResNet-M + SE (primeras 3 etapas + SE final) = el "-A".

Todos exponen forward(x, return_logits=False) para ser intercambiables con
TResNetLA en train.py::train_model.
"""

import torch
import torch.nn as nn
import torchvision

from model import SEBlock, build_model


class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, se: bool = False):
        super().__init__()
        net = torchvision.models.resnet50(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                              padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net
        self.se = se
        if se:
            self.se1 = SEBlock(256, reduction=16)
            self.se2 = SEBlock(512, reduction=16)
            self.se3 = SEBlock(1024, reduction=16)
            self.se4 = SEBlock(2048, reduction=16)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.maxpool(x)
        x = n.layer1(x); x = self.se1(x) if self.se else x
        x = n.layer2(x); x = self.se2(x) if self.se else x
        x = n.layer3(x); x = self.se3(x) if self.se else x
        x = n.layer4(x); x = self.se4(x) if self.se else x
        x = n.avgpool(x).flatten(1)
        logits = n.fc(x)
        return logits if return_logits else torch.sigmoid(logits)


TRESNET_M_DEPTHS = (3, 4, 11, 3)


def build_baseline(name: str, cfg, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        return ResNet50MultiLabel(num_classes, se=False)
    if name in ("resnet50-a", "resnet50a"):
        return ResNet50MultiLabel(num_classes, se=True)
    if name in ("tresnet-m", "tresnet_m"):
        from model import TResNetLA
        return TResNetLA(num_classes, in_channels=1,
                         stage_depths=TRESNET_M_DEPTHS,
                         final_se_reduction=cfg.FINAL_SE_REDUCTION, use_se=False)
    if name in ("tresnet-ma", "tresnet_ma"):
        from model import TResNetLA
        return TResNetLA(num_classes, in_channels=1,
                         stage_depths=TRESNET_M_DEPTHS,
                         final_se_reduction=cfg.FINAL_SE_REDUCTION, use_se=True)
    raise ValueError(f"baseline desconocido: {name}")
