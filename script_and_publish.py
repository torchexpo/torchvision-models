"""Driver Program"""
import os
from typing import Any, List

import gaama
import torch
import torchvision
from torchvision.io.image import read_image

from tasks import ImageClassificationModule, SemanticSegmentationModule, ObjectDetectionModule


SKIPPED_MODELS = ['googlenet', 'inception_v3']


def get_models_and_weights(module: Any) -> Any:
    """Get torchvision models and weights"""
    all_models = torchvision.models.list_models(module=module)
    for skipped_model in SKIPPED_MODELS:
        if skipped_model in all_models:
            all_models.remove(skipped_model)
    return all_models


def publish_release(version: str, files: List[str]) -> None:
    """Create GitHub Release using GaAMA"""
    print("github release:", version)
    print("release assets:", files)
    release = gaama.GaAMA(
        username=os.getenv("GIT_USERNAME"),
        password=os.getenv("GIT_PASSWORD"),
        owner="torchexpo",
        repository="torchvision-models")
    release.publish(tag=version, files=files, zip_files=False)


def script_and_save(model: str, weight: Any, example: Any, is_detection: bool) -> str:
    """Script the model, save the output and select for publish"""
    slug = str(model).replace("_", "-").lower() + "-" + \
        str(weight).split(".", 1)[1].replace("_", "-").lower()
    filename = slug+".pt"
    try:
        derived_model = torchvision.models.get_model(
            model, weights=weight, progress=False)
    except Exception as exp:  # pylint: disable=broad-except
        print(str(exp))
        print("error deriving model:", slug)
    scripted_module = torch.jit.script(
        script_module(model=derived_model))
    scripted_module.save(filename)
    # delete weights
    os.remove(os.path.expanduser('~') +
              "/.cache/torch/hub/checkpoints/"+weight.url.split("/")[-1:][0])
    try:
        run_example(filename, weight, example, is_detection)
        return filename
    except Exception as exp:  # pylint: disable=broad-except
        print(str(exp))
        print("error running example for model:", slug)
    return None


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
    task_and_script_modules = []
    if os.getenv("TASK") == "image-classification":
        task_and_script_modules.append([torchvision.models, ImageClassificationModule,
                                        "examples/image_classification.jpg", False])
    elif os.getenv("TASK") == "semantic-segmentation":
        task_and_script_modules.append([torchvision.models.segmentation, SemanticSegmentationModule,
                                        "examples/semantic_segmentation.jpg", False])
    elif os.getenv("TASK") == "object-detection":
        task_and_script_modules.append([torchvision.models.detection, ObjectDetectionModule,
                                        "examples/object_detection.jpg", True])
    output_files = []
    items = []
    for item in task_and_script_modules:
        task_module, script_module, _example, _is_detection = item
        models = get_models_and_weights(task_module)
        for _model in models:
            weights = torchvision.models.get_model_weights(_model)
            for _weight in weights:
                result = script_and_save(
                    _model, _weight, _example, _is_detection)
                if result is not None:
                    output_files.append(result)
    publish_release(os.getenv("VERSION")+"-"+os.getenv("TASK"), output_files)
