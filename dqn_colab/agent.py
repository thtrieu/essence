from src.net import Net
from src.utils.misc import uniform, xavier, perm
from .qfunction import Qfunction
import numpy as np

class Agent(object):
    
    ''' each action is a pair (direct, signal)'''
    _ACTION_SPACE = [[0., 0.], [0., 1.],
                     [1., 0.], [1., 1.]]

    def __init__(self, mlp_dims,
                 batch = 32, gamma = .95):
        self._batch = batch
        self._gamma = gamma
        self._exreplay = list() # experience replay
        self._moving_q = Qfunction(False, *mlp_dims)
        self._target_q = Qfunction(True, *mlp_dims)
    
    def _best_act(self, qfunc, observe):
        max_q_out = None
        observe = list(observe)
        for action in self._ACTION_SPACE:
            q_inp = np.array([observe + action])
            q_out = qfunc.forward(q_inp)

            if not max_q_out or max_q_out < q_out: 
                max_q_out = q_out
                best_act = action
                
        return best_act, max_q_out

    def act(self, observe, epsilon):
        rand = uniform()
        if rand <= epsilon:
            direct = round(uniform())
            signal = round(uniform())
            q_val = None
        else:
            (direct, signal), q_val = \
                self._best_act(
                    self._moving_q, observe)
        return direct, signal
    
    def store_and_learn(self, transition):
        self._exreplay.append(transition)

        shuffle = perm(len(self._exreplay))
        mini_batch = [self._exreplay[i] \
            for i in shuffle[:self._batch]]

        observe_action_t, \
        reward_t        , \
        observe_tplus1  , \
            = map(np.array, list(zip(*mini_batch)))

        best_q_tplus1 = list(map(
            lambda x: self._best_act(self._target_q, x)[1],
            observe_tplus1 
        ))

        best_q_tplus1 = np.array(best_q_tplus1)[:, None]
        best_q_tplus1 *= self._gamma
        reward_t += best_q_tplus1
        loss = self._moving_q.train(
            observe_action_t, reward_t)
    
    def update(self):
        self._target_q.assign(
            self._moving_q.yield_params_values())