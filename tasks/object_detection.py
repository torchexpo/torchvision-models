"""Object Detection TorchScript Module"""
from typing import List, Dict

import torch


class ObjectDetectionModule(torch.nn.Module):
    """Object Detection TorchScript Module"""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, model_input: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Module forward"""
        output = self.model([model_input])
        return output[1]
