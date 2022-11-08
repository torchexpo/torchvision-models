"""Semantic Segmentation TorchScript Module"""
from typing import Any

import torch


class SemanticSegmentationModule(torch.nn.Module):
    """Semantic Segmentation TorchScript Module"""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, model_input: torch.Tensor) -> Any:
        """Module forward"""
        output = self.model(model_input)
        prediction = output["out"]
        normalized_masks = prediction.softmax(dim=1)
        return normalized_masks
