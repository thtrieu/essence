import numpy as np
import cv2

class Environment(object):
    
    _w = _h = 256
    _canvas = np.ones((_w, _h, 3)) 
    _canvas *= 255.

    def __init__(self):
        pass
    
    def react(self):
        pass
    
    def appearance(self):
        pass

    def _reset(self):
        pass

    def viz(self, waitKey = 1):
        self.visualize()
        return cv2.waitKey(waitKey)