from typing import Any, Callable, Tuple, List, Union

from liga.mixin import Pretrained
from liga.registry.model import ModelType, ModelSpec
from ligavision.spark.types import Box2d

def convert_pred_groups_to_box2d(pred_groups):
    result_groups = []
    for i in range(len(pred_groups)):
        pred_group = pred_groups[i]
        result_group = []
        for pred in pred_group:
            points = pred[0]
            text = pred[1]
            conf = pred[2]
            point_x = []
            point_y = []
            for point in points:
                point_x.append(point[0])
                point_y.append(point[1])
            bbox = Box2d(min(point_x), min(point_y), max(point_x), max(point_y))
            result = {'text': text.lower(), 'bbox': bbox}
            result_group.append(result)
        result_groups.append(result_group)
    return result_groups


class EasyOCRModelType(ModelType, Pretrained):
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self, spec: ModelSpec, **kwargs):
        self.model = self.pretrained_model()

    def pretrained_model(self):
        import easyocr
        return easyocr.Reader(['en'])
    
    def schema(self) -> str:
        return "array<struct<text:string,bbox:box2d>>"

    def transform(self) -> Callable:
        return lambda image: image.to_numpy()

    def predict(self, images, *args, **kwargs) -> Any:
        pred_groups = []
        for image in images:
            pred_group = self.model.readtext(image, canvas_size=600)
            pred_groups.append(pred_group)
        return convert_pred_groups_to_box2d(pred_groups)

MODEL_TYPE = EasyOCRModelType()
