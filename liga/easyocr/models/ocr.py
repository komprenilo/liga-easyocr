from typing import Any, Callable, Tuple

from numpy import ndarray
import easyocr

from liga.mixin import Pretrained
from liga.registry.model import ModelType, ModelSpec
from ligavision.spark.types import Mask
from ligavision.dsl import Image


def convert_pred_groups_for_rikai(pred_groups, shapes):
    result_groups = []
    for i in range(len(pred_groups)):
        pred_group = pred_groups[i]
        shape = shapes[i]
        result_group = []
        for pred in pred_group:
            points = pred[0]
            text = pred[1]
            poly = []
            for point in points:
                poly.append(float(point[0]))
                poly.append(float(point[1]))
            mask = Mask.from_polygon([poly], shape[1], shape[0])
            result = {'text': text, 'mask': mask}
            result_group.append(result)
        result_groups.append(result_group)
    return result_groups


class EasyOCRModelType(ModelType, Pretrained):
    def __init__(self, language):
        super().__init__()
        self.model = None
        if isinstance(language, list):
            self.langs = language
        elif isinstance(language, str):
            self.langs = [language]
        else:
            raise RuntimeError("Invalid type of language")

    def load_model(self, spec: ModelSpec, **kwargs):
        self.model = self.pretrained_model()

    def pretrained_model(self):
        #  https://www.jaided.ai/easyocr/
        return easyocr.Reader(self.langs)
    
    def schema(self) -> str:
        return "array<struct<text:string,mask:mask>>"

    def transform(self) -> Callable:
        return lambda image: image.to_numpy()

    def predict(self, images, *args, **kwargs) -> Any:
        def _ndarray_to_shape(image: ndarray) -> Tuple:
            pil_image = Image.from_array(image).to_pil()
            return (pil_image.width, pil_image.height)

        pred_groups = []
        for image in images:
            pred_group = self.model.readtext(image, canvas_size=600)
            pred_groups.append(pred_group)
        shapes = [_ndarray_to_shape(image) for image in images]
        return convert_pred_groups_for_rikai(pred_groups, shapes)

MODEL_TYPE = EasyOCRModelType(["en"])
