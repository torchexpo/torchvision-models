"""Driver Program"""
import os
from typing import Any, List

from PIL import Image
import torch
import torchvision

from image_classification import ImageClassificationModule


def get_models_and_weights(module: Any) -> Any:
    """Get torchvision models and weights"""
    return torchvision.models.list_models(
        module=module)


def publish_release(version: str, files: List[str]) -> None:
    """Create GitHub Release using GaAMA"""
    print("github release:", version)
    print("release assets:", files)


def run_example(module: str, model_weight: Any, example_img: Any) -> None:
    """Run example inference by loading scripted module"""
    preprocess = model_weight.transforms()
    model_input = preprocess(Image.open(example_img))
    loaded_module = torch.jit.load(module)
    loaded_module.forward(model_input.unsqueeze(0))


if __name__ == '__main__':
    print('torch:', torch.__version__)
    print('torchvision:', torchvision.__version__)
    task_and_script_modules = [[torchvision.models, ImageClassificationModule,
                                "examples/image_classification.jpg"],
                               [torchvision.models.segmentation, None, ""],
                               [torchvision.models.detection, None, ""]]
    output_files = []
    for item in task_and_script_modules:
        task_module, script_module, example = item
        if script_module is not None:
            models = get_models_and_weights(task_module)
            for model in models:
                weights = torchvision.models.get_model_weights(model)
                for weight in weights:
                    slug = str(model).replace("_", "-").lower() + "-" + \
                        str(weight).split(".", 1)[1].replace("_", "-").lower()
                    derived_model = torchvision.models.get_model(
                        model, weights=weight)
                    scripted_module = torch.jit.script(
                        script_module(model=derived_model))
                    filename = slug+".pt"
                    scripted_module.save(filename)
                    output_files.append(filename)
                    try:
                        run_example(filename, weight, example)
                    except:  # pylint: disable=bare-except
                        print("some error running example for model:", slug)
                    break
                break
    publish_release(os.getenv("VERSION"), output_files)
