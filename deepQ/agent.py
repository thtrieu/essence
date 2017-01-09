from src.net import Net
from src.utils.misc import uniform, xavier, perm
from .qfunction import Qfunction
import numpy as np

class Agent(object):
    def __init__(self, mlp_dims, action_space,
                 batch = 32, gamma = .987):
        self._batch = batch
        self._gamma = gamma
        self._action_space = action_space
        self._exreplay = list() # experience replay
        self._moving_q = Qfunction(False, *mlp_dims)
        self._target_q = Qfunction(True, *mlp_dims)
    
    def _best_act(self, qfunc, observe):
        max_q_out = None
        observe = list(observe)
        for action in self._action_space:
            q_inp = np.array([observe + action])
            q_out = qfunc.forward(q_inp)

            if not max_q_out or max_q_out < q_out: 
                best_act = action
                max_q_out = q_out
                
        return best_act, max_q_out

    def act(self, observe, epsilon):
        rand = uniform()
        if rand <= epsilon: # exploration
            best_act = None
            q_val = None
        elif rand > epsilon: # exploitation
            best_act, q_val = \
                self._best_act(
                    self._moving_q, observe)
        return best_act, q_val
    
    def store_and_learn(self, transition):
        self._exreplay.append(transition)

        shuffle = perm(len(self._exreplay))[:self._batch]
        mini_batch = [self._exreplay[i] for i in shuffle]

        observe_action_t, \
        reward_t        , \
        observe_tplus1  , \
            = map(np.array, zip(*mini_batch))

        best_q_tplus1 = np.array(list(map(
            lambda x: self._best_act(self._target_q, x)[1],
            observe_tplus1 
        )))

        reward_t += best_q_tplus1 * self._gamma
        loss = self._moving_q.train(
            observe_action_t, reward_t[:, None])
    
    def update(self):
        self._target_q.assign(
            self._moving_q.yield_params_values())
    
    def save(self, file_name):
        self._target_q.save(file_name + '_target')
        self._moving_q.save(file_name + '_moving')
    
    def load(self, file_name):
        self._target_q.load(file_name + '_target')
        self._moving_q.load(file_name + '_moving')