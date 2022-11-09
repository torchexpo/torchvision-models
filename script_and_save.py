"""Driver Program"""
import os
from typing import Any, List

import gaama
import torch
import torchvision
from torchvision.io.image import read_image

from tasks import ImageClassificationModule, SemanticSegmentationModule, ObjectDetectionModule


def get_models_and_weights(module: Any) -> Any:
    """Get torchvision models and weights"""
    return torchvision.models.list_models(module=module)


def publish_release(version: str, files: List[str]) -> None:
    """Create GitHub Release using GaAMA"""
    print("github release:", version)
    print("release assets:", files)
    release = gaama.GaAMA(
        username=os.getenv("GIT_USERNAME"),
        password=os.getenv("GIT_PASSWORD"),
        owner="torchexpo",
        repository="torchvision-models")
    release.publish(tag=os.getenv("VERSION"), files=files, zip_files=False)


def run_example(module: str, model_weight: Any, example_img: Any, detection: bool = False) -> None:
    """Run example inference by loading scripted module"""
    preprocess = model_weight.transforms()
    img = read_image(example_img)
    model_input = preprocess(img)
    loaded_module = torch.jit.load(module)
    if detection:
        loaded_module.forward(model_input)
    else:
        loaded_module.forward(model_input.unsqueeze(0))


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    task_and_script_modules = [
        [torchvision.models, ImageClassificationModule,
         "examples/image_classification.jpg", False],
        [torchvision.models.segmentation, SemanticSegmentationModule,
         "examples/semantic_segmentation.jpg", False],
        [torchvision.models.detection, ObjectDetectionModule,
         "examples/object_detection.jpg", True]
    ]
    output_files = []
    for item in task_and_script_modules:
        task_module, script_module, example, is_detection = item
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
                        run_example(filename, weight, example, is_detection)
                    except Exception as e:  # pylint: disable=broad-except
                        print(str(e))
                        print("some error running example for model:", slug)
                    break
                break
    publish_release(os.getenv("VERSION"), output_files)
