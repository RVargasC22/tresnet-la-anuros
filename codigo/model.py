"""
Arquitectura TResNet-LA: ResNet (Basic/Bottleneck blocks) + atención de canal
(Squeeze-and-Excitation) para reconocimiento multi-etiqueta de audio bajo
interferencia, tal como se describe en el paper (Sección 4, Fig. 7, Tabla C1).

Componentes:
    - SpaceToDepth: reorganiza bloques espaciales del espectrograma en canales.
    - SEBlock: módulo de atención de canal (squeeze -> excitation -> scale).
    - BasicBlockSE: bloque residual simple (2x conv3x3) + SE.
    - BottleneckBlockSE: bloque residual cuello de botella (1x1-3x3-1x1) + SE.
    - TResNetLA: red completa, salida Sigmoid multi-label.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SpaceToDepth
# ---------------------------------------------------------------------------
class SpaceToDepth(nn.Module):
    """Reorganiza bloques espaciales de tamaño (block_size x block_size) en el
    eje de canales. Reduce la resolución espacial y aumenta los canales,
    equivalente a un stride convolucional pero sin pérdida de información."""

    def __init__(self, block_size: int = 4):
        super().__init__()
        self.bs = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        bs = self.bs
        x = x.view(n, c, h // bs, bs, w // bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * bs * bs, h // bs, w // bs)
        return x


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (channel attention)
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Bloque de atención de canal (Hu et al., 2018), Ecuaciones (14)-(16) del
    paper:
        Squeeze:   Zc = GlobalAvgPool(X)                       -> (N, C, 1, 1)
        Excite:    F  = sigmoid(W2 * relu(W1 * Zc))            -> (N, C, 1, 1)
        Scale:     X_hat = F * X                                (canal a canal)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = x.shape
        z = self.avg_pool(x).view(n, c)          # squeeze
        f = self.fc(z).view(n, c, 1, 1)           # excitation
        return x * f                              # scale (reweighting de canal)


# ---------------------------------------------------------------------------
# IBN (Inplace-BatchNorm): en este paper se usa como BatchNorm estándar,
# manteniendo el nombre del paper por claridad conceptual.
# ---------------------------------------------------------------------------
def ibn(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels)


# ---------------------------------------------------------------------------
# BasicBlock con SE (Fig. 7, subplot central)
# ---------------------------------------------------------------------------
class BasicBlockSE(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, use_se=True,
                 se_reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = ibn(out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = ibn(out_ch)
        self.se = SEBlock(out_ch, reduction=se_reduction) if use_se else nn.Identity()
        self.downsample = downsample
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu_out(out)


# ---------------------------------------------------------------------------
# BottleneckBlock con SE (Fig. 7, subplot derecho)
# ---------------------------------------------------------------------------
class BottleneckBlockSE(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None, use_se=True,
                 se_reduction=8):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = ibn(mid_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = ibn(mid_ch)
        self.se = SEBlock(mid_ch, reduction=se_reduction) if use_se else nn.Identity()
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = ibn(out_ch)
        self.downsample = downsample
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu_out(out)


def _make_downsample(in_ch, out_ch, stride):
    """Downsample por average pooling + conv1x1 (igual que TResNet original)."""
    layers = []
    if stride > 1:
        layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True))
    layers.append(nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False))
    layers.append(ibn(out_ch))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# TResNet-LA completo
# ---------------------------------------------------------------------------
class TResNetLA(nn.Module):
    """
    Implementación siguiendo Fig. 7 y Tabla C1 del paper (anchos 76/152/304/608,
    ~53 M parámetros, = TResNet-L):

        Input (1, T, F)
          -> SpaceToDepth(4)
          -> Conv1x1 (stem, 76 canales)
          -> BasicBlockSE   x stage_depths[0]   canal 76    (con SE, r=4)
          -> BasicBlockSE   x stage_depths[1]   canal 152   (con SE, r=4, downsample x2)
          -> BottleneckSE   x stage_depths[2]   mid 304/out 1216  (con SE, r=8, downsample x2)
          -> BottleneckSE   x stage_depths[3]   mid 608/out 2432  (sin SE,      downsample x2)
          -> SE final (r=16, C152)
          -> GlobalAvgPool -> FC -> Sigmoid -> (N, num_classes)
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 stage_depths=(4, 5, 18, 3), stage_use_se=(True, True, True, False),
                 base_width: int = 76, final_se_reduction: int = 16,
                 space_to_depth_block: int = 4, use_se: bool = True,
                 stage_width_mult=(1, 2, 4, 8)):
        # stage_width_mult: multiplicadores de base_width para el canal de cada
        # etapa (pre-expansion del Bottleneck). (1,2,4,8) reproduce la Tabla C1
        # del paper: 76, 152, 304 (out 1216), 608 (out 2432).
        super().__init__()
        # use_se=False desactiva TODA atención de canal (equivalente a
        # TResNet-L puro, sin SE en ningún lado) -- se usa para la ablación
        # que aísla la contribución del módulo SE (Sección 5.2/5.5 del paper).
        if not use_se:
            stage_use_se = (False, False, False, False)
        self.use_se = use_se

        self.space_to_depth = SpaceToDepth(space_to_depth_block)
        stem_in_ch = in_channels * space_to_depth_block * space_to_depth_block

        # Tabla C1 (texto) dice "Conv1: 3x3, 76, stride 2", pero esto es
        # inconsistente con su propia columna de Output Size (125x32, igual a
        # la fila anterior de SpaceToDepth -- un stride 2 la reduciria) y con
        # el texto narrativo ("este conv solo ajusta el numero de canales").
        # La Fig. 7 (diagrama dibujado, mas confiable que la tabla) resuelve
        # la ambiguedad de forma explicita: "Conv 1x1". Se usa esa version.
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in_ch, base_width, kernel_size=1, stride=1, bias=False),
            ibn(base_width),
            nn.LeakyReLU(0.01, inplace=True),
        )

        w = [base_width * m for m in stage_width_mult]   # (76, 152, 304, 608)
        self.in_ch = base_width
        # Stage 1: BasicBlock, sin downsample espacial
        self.stage1 = self._make_stage(
            BasicBlockSE, out_ch=w[0], blocks=stage_depths[0],
            stride=1, use_se=stage_use_se[0], se_reduction=4,
        )
        # Stage 2: BasicBlock, downsample x2
        self.stage2 = self._make_stage(
            BasicBlockSE, out_ch=w[1], blocks=stage_depths[1],
            stride=2, use_se=stage_use_se[1], se_reduction=4,
        )
        # Stage 3: BottleneckBlock (expansion x4), downsample x2
        self.stage3 = self._make_stage(
            BottleneckBlockSE, out_ch=w[2], blocks=stage_depths[2],
            stride=2, use_se=stage_use_se[2], se_reduction=8,
        )
        # Stage 4: BottleneckBlock (expansion x4), downsample x2, sin SE interno
        self.stage4 = self._make_stage(
            BottleneckBlockSE, out_ch=w[3], blocks=stage_depths[3],
            stride=2, use_se=stage_use_se[3], se_reduction=8,
        )

        # SE layer final (r=16), aplicada sobre el feature map de salida.
        # Se omite por completo si use_se=False (modelo de ablación sin
        # atención de canal en absoluto).
        self.final_se = SEBlock(self.in_ch, reduction=final_se_reduction) if use_se \
            else nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_ch, num_classes)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _make_stage(self, block_cls, out_ch, blocks, stride, use_se, se_reduction):
        downsample = None
        expansion = block_cls.expansion
        if stride != 1 or self.in_ch != out_ch * expansion:
            downsample = _make_downsample(self.in_ch, out_ch * expansion, stride)

        layers = [block_cls(self.in_ch, out_ch, stride=stride, downsample=downsample,
                             use_se=use_se, se_reduction=se_reduction)]
        self.in_ch = out_ch * expansion
        for _ in range(1, blocks):
            layers.append(block_cls(self.in_ch, out_ch, stride=1, downsample=None,
                                     use_se=use_se, se_reduction=se_reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        # x: (N, 1, T, F)  ->  T ~= 500 frames temporales, F = 128 bandas Mel
        x = self.space_to_depth(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_se(x)
        x = self.global_pool(x).flatten(1)
        logits = self.fc(x)
        if return_logits:
            return logits
        return self.sigmoid(logits)


def build_model(num_classes: int, cfg, use_se: bool = True) -> TResNetLA:
    """use_se=False construye el modelo de ablación (sin atención de canal
    en absoluto), manteniendo exactamente la misma profundidad/arquitectura
    base para que la comparación sea justa."""
    return TResNetLA(
        num_classes=num_classes,
        in_channels=1,
        stage_depths=cfg.STAGE_DEPTHS,
        stage_use_se=cfg.STAGE_USE_SE,
        final_se_reduction=cfg.FINAL_SE_REDUCTION,
        use_se=use_se,
    )


if __name__ == "__main__":
    # Sanity check rápido: forward pass con tensor dummy.
    import config as cfg
    model = build_model(num_classes=42, cfg=cfg)
    dummy = torch.randn(2, 1, cfg.TARGET_TIME_FRAMES, cfg.N_MELS)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print("Output shape:", out.shape)          # esperado: (2, 42)
    print("Output range:", out.min().item(), out.max().item())
    print(f"Total parámetros: {n_params:,}")
