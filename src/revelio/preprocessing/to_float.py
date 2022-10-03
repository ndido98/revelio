import numpy as np

from revelio.dataset.element import ElementImage

from .step import PreprocessingStep


class ToFloat(PreprocessingStep):
    def process_element(self, elem: ElementImage) -> ElementImage:
        if elem.image.dtype == np.uint8:
            new_img = elem.image.astype(np.float32) / 255.0
            return ElementImage(
                path=elem.path,
                image=new_img,
                landmarks=elem.landmarks,
                features=elem.features,
            )
        else:
            return elem
