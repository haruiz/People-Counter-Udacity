from utils import Vis_Util
import numpy as np


class Box:
    def __init__(self, xmin, ymin, xmax, ymax, score, label_id, mask=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.label_id = label_id
        self.mask = mask

    def draw(self, image, label=None, color: tuple = (255, 0, 255)):
        Vis_Util.draw_box(image, self.xmin, self.ymin, self.xmax, self.ymax, label, color)
        if isinstance(self.mask, np.ndarray):
            Vis_Util.draw_mask(image, self.mask, color)