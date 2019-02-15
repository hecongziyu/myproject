import math
import numpy as np
import sys
import pdb
EPS = 1e-8

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


    def getActionProb(self, playerTable, cur_play, action=0, temp=1):
        cur_play = 0
        states = self.game.getTableStates(playerTable,action)
        # print('states --> {}'.format(states))
        # p,v = self.nnet.predict(s)
        valid_actions = self.game.getValidActions(playerTable,cur_play, action)
        # play table , action 上一个player的action
        for i in range(20):
            self.search(playerTable, action)
            # r = input(". Continue? [y|n]")
            # if r != "y":
            #     sys.exit()            

        s = self.game.stringRepresentation(states)

        # 根据MCTS的s,a状态次数，来选择action
        # print('Counts Nsa {}'.format(self.Nsa))
        # print('Counts state s  {}'.format(s))

        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        # print('couns {}'.format(counts))
        # 需修改成树搜索方式，本处暂时不要
        if temp==0:
            bestA = np.argmax(valid_actions)
            probs = [0]*len(valid_actions)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]

        # probs = p * valid_actions

        # probs = [x/float(sum(probs)) for x in probs]
        return probs



    def search(self, playerTable,action):
        # print('search play table {} action {}'.format(playerTable, action))
        # print('Qsa {}'.format(self.Qsa))
        # print('Nsa {}'.format(self.Nsa))
        # print('Es {}'.format(self.Es))
        # print('Ps {}'.format(self.Ps))
        # pdb.set_trace()
        
        states = self.game.getTableStates(playerTable,action)
        s = self.game.stringRepresentation(states)

        # ES  stores game.getGameEnded ended for board s
        if s not in self.Es:
            self.Es[s] = self.game.checkGameEnded(playerTable, 0, action)

        if self.Es[s][0]!=0:
            # terminal node
            return self.Es[s][0]

        # PS stores initial policy (returned by neural net)

        if s not in self.Ps:

            # leaf node
            self.Ps[s], v = self.nnet.predict(states)
            valids = self.game.getValidActions(playerTable,0, action)
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
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # print('table {} states {} valids actions {}'.format(playerTable,s, valids))
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

        next_s, next_player = self.game.getNextState(playerTable,0,a)
        # print('cur table {} next table {} next action {}'.format(playerTable, next_s, a))
        # next_s = self.game.getTableFrom(playerTable, next_player)

        v = self.search(next_s, a)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v




    


