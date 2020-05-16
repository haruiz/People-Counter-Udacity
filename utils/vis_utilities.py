import numpy as np
from PIL import Image
import cv2


class Vis_Util:
    @staticmethod
    def draw_mask(image, mask, color=(255, 0, 0), alpha=0.4):
        assert image.dtype == np.uint8, '`image` not of type np.uint8'
        assert mask.dtype == np.uint8, '`mask` not of type np.uint8'
        if np.any(np.logical_and(mask != 1, mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        if image.shape[:2] != mask.shape:
            raise ValueError('The image has spatial dimensions %s but the mask has '
                             'dimensions %s' % (image.shape[:2], mask.shape))
        pil_image = Image.fromarray(image)
        solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(color), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        np.copyto(image, np.array(pil_image.convert('RGB')))

    @staticmethod
    def draw_box(img, xmin, ymin, xmax, ymax, label=None, color=(255, 0, 0)):
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        if label:
            cv2.putText(img, label, (xmin - 10, ymax - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
