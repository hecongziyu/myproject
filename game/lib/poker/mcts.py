import math
import numpy as np


class MCTS(object):
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s


    def getActionProb(self, states, cur_play, action, temp=1):
        # s = states
        # p,v = self.nnet.predict(s)
        # valid_actions = self.game.getValidActions(cur_play, action)

        for i in range(10):
            self.search(states, cur_play, action)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        # 需修改成树搜索方式，本处暂时不要
        if temp==0:
            bestA = np.argmax(valid_actions)
            probs = [0]*len(valid_actions)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs



    def search(self, states,cur_play, action):
        s = self.game.stringRepresentation(states)

        # ES  stores game.getGameEnded ended for board s
        if s not in self.Es:
            self.Es[s] = self.game.checkGameEnded((cur_play+1)%2, action)
        if self.Es[s][0]!=0:
            # terminal node
            return [-x for x in self.Es[s]]

         # PS stores initial policy (returned by neural net)
         if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(states)
            valids = self.game.getValidMoves((cur_play+1)%2, action)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                # 如果所有有效移动都被屏蔽，则使所有有效移动的可能性相等
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            # Ns stores #times board s was visited
            self.Ns[s] = 0
            #  ？？？ why return -v
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + 1*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = 1*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(cur_play, a)
        next_s = self.game.getTableFrom(next_player)
        next_s = self.game.getTableStates(next_s)
        v = self.search(next_s, next_player, a)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v




    


