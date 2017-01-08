from src.utils.misc import uniform
import numpy as np
import cv2

class Slider(object):
    def __init__(self):
        self._trace = list()
        self._reset()
    
    @property
    def ball_colour(self):
        return self._colour

    def react(self, control_left, control_right):
        self._slide = (control_left, control_right)
        acceleration = control_left - control_right
        self._ball_x += 1e-3 * self._ball_v
        self._ball_v += .1 * acceleration
        
        x = self._ball_x
        if x < 0. or x > 1.:
            if (0.5 - x) * (0.5 - self._colour) > 0.:
                self._trace.append(str(self._colour))
            else:
                self._trace.append('_')
            self._reset()

        c = self._colour
        reward = (c * 2. - 1.) * (x - .5)
        return reward

    def _reset(self):
        self._ball_x = 0.5
        self._ball_v = 0.0
        self._colour = round(uniform())
        trace50 = ''.join(self._trace[-50:])
        if self._colour: print('>>>', trace50)
        else: print('<<<', trace50)

    def visualize(self, mes1, mes2):
        canvas = np.ones((512, 512, 3)) * 255.
        ball_x = int(self._ball_x * 512)
        colour = [0, 0, 0]
        colour[int(self._colour)*2] = 255.
        canvas = cv2.circle(
            canvas, (ball_x, 256), 10, colour)
        cv2.imshow('', canvas)