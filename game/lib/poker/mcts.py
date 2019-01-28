import math
import numpy as np


class MCTS(object):
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

    def getActionProb(self, states, cur_play, action, temp=1):
        s = states
        p,v = self.nnet.predict(s)
        valid_actions = self.game.getValidActions(cur_play, action)

        # 需修改成树搜索方式，本处暂时不要
        # if temp==0:
        #     bestA = np.argmax(valid_actions)
        #     probs = [0]*len(valid_actions)
        #     probs[bestA]=1
        #     return probs

        p = p*valid_actions
        p = [x/float(sum(p)) for x in p]
        return p
    


