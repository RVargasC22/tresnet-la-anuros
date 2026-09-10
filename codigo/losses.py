"""
Asymmetric Loss (ASL) para clasificación multi-etiqueta (Ridnik, Ben-Baruch,
et al., 2021), usada en el paper con gamma_neg=6, gamma_pos=0 (Sección 5.1).

Intuición: en tareas multi-label, la mayoría de las etiquetas son negativas
(0) para una muestra dada, lo que domina el gradiente. ASL down-weightea los
negativos fáciles (gamma_neg alto) y deja casi intactos los positivos
(gamma_pos=0), acelerando la convergencia frente al desbalance positivo/
negativo que describe el paper (Sección 3.1 y 3.2).
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 6, gamma_pos: float = 0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: salida CRUDA de la red (antes de sigmoid), shape (N, C)
        targets: etiquetas binarias {0,1}, shape (N, C)
        """
        # Forzar fp32 aunque el entrenamiento use mixed precision (autocast):
        # el log() de la ASL con probabilidades cercanas a 0 pierde precision
        # en fp16 y produce NaN. Esta linea era la causa del loss=NaN loggeado.
        logits = logits.float()
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # probability shifting para negativos (margen de tolerancia)
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))

        # focusing asimétrico: distinto gamma para positivos y negativos
        with torch.no_grad():
            pt_pos = probs_pos * targets
            pt_neg = probs_neg * (1 - targets)
            pt = pt_pos + pt_neg
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            focusing_weight = torch.pow(1 - pt, gamma)

        loss = (loss_pos + loss_neg) * focusing_weight
        return -loss.sum(dim=1).mean()