import cv2
import numpy as np

from mesh_utils import mls_affine_deformation, mls_similarity_deformation, mls_rigid_deformation

class BodyShapeTransfer():
    def __init__(self, control, oriShape, dstShape):
        # Shape: Dictionary(height, BMI, type)
        self.p = control
        self.oriShape = oriShape
        self.dstShape = dstShape

        self.q = self.p

        return

    def transform(self, input_path):
        oriImg = cv2.imread(input_path)
        height, width, _ = oriImg.shape
        gridX = np.arange(width, dtype=np.int16)
        gridY = np.arange(height, dtype=np.int16)
        vy, vx = np.meshgrid(gridX, gridY)

        rigid = mls_rigid_deformation(vy, vx, self.p, self.q, alpha=1)
        dstImg = np.ones_like(oriImg)
        dstImg[vx, vy] = oriImg[tuple(rigid)]
        return dstImg