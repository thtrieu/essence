from src.utils.misc import uniform
from .env import Environment
import numpy as np
import cv2

'''
accelerations calculations are from eq. 23 & 24 of:
http://coneural.org/florian/papers/05_cart_pole.pdf
'''

_G = 9.8
_F = 15.0
_MCART = 1.0
_MPOLE = 0.3
_MALL = _MCART + _MPOLE
_LEN = 0.7
_POLE_MLEN = _MPOLE * _LEN
_ALPHAX = 1e-2
_ALPHAA = 2e-2
_THRES = .3

class CartPole(Environment):
    def __init__(self):
        self._dead = -1
        self._reset()

    def react(self, control):
        cos = np.cos(self._angle)
        sin = np.sin(self._angle)
        tmp = _POLE_MLEN * (self._vangle**2) * sin
        tmp = (tmp + control[0] * _F) / _MALL

        #Angular and position accelerations
        aangle =  (_G * sin - cos * tmp)
        aangle /= _LEN * (4./3. - _MPOLE * (cos**2) / _MALL)
        accele = tmp - _POLE_MLEN * aangle * cos / _MALL
        
        self._x = self._x + _ALPHAX * self._v
        self._v = self._v + _ALPHAX * accele
        self._angle = self._angle + _ALPHAA * self._vangle
        self._vangle = self._vangle + _ALPHAA * aangle

        reward = - abs(self._angle)
        if self._violates(): self._reset()

        return reward
    
    def appearance(self):
        return [self._x, self._v, self._angle, self._vangle]

    def _violates(self):
        check1 = self._angle < - _THRES
        check2 = self._angle > + _THRES
        check3 = self._x < -1.0
        check4 = self._x > +1.0
        return check1 or check2 or check3 or check4

    def _reset(self):
        self._x = 0.0
        self._v = 0.0
        self._angle = 0.0
        self._vangle = 0.0
        self._dead += 1

    def visualize(self):
        # img = self._canvas.copy()
        self._w = 1024; self._h = 256
        img = np.ones((self._h, self._w, 3)) * 255.
        centerx = (self._x * .5 + .5) * self._w
        centery = 0.75 * self._h
        
        upleft = (int(centerx - 10), int(centery - 5))
        botright = (int(centerx + 10), int(centery + 5))
        img = cv2.rectangle(img, upleft, botright, (255, 0, 0), 3)

        start = (int(centerx), int(centery))
        end = (int(centerx + 200 * _LEN * np.sin(self._angle)),
               int(centery - 200 * _LEN * np.cos(self._angle)))
        img = cv2.line(img, start, end, (0, 0, 255))
        message = 'Dead: {}'.format(self._dead)
        img = cv2.putText(img, message, (10, self._h - 10), 
                          0, .4, (0, 0, 0), 1)
        cv2.imshow('cart-pole', img)