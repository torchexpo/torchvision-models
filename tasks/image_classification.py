"""Image Classification TorchScript Module"""
from typing import Any

import torch


class ImageClassificationModule(torch.nn.Module):
    """Image Classification TorchScript Module"""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, model_input: torch.Tensor) -> Any:
        """Module forward"""
        output = self.model(model_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities
