import math
import numpy as np
import sys
import pdb
import copy
EPS = 1e-8

class MCTS(object):
    def __init__(self, nnet, args):
        # self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s


    def getActionProbByRandom(self,game,cur_play,action, temp=1):
        validActions = game.getValidActions(cur_play, action)
        bestA = np.argmax(validActions)
        probs = [0]*len(validActions)
        probs[bestA]=1
        return probs        

    def getActionProb(self, game, cur_play, action=0, temp=1):
        states = game.getTableStates(cur_play,action)
        # print('states --> {}'.format(states))
        # p,v = self.nnet.predict(s)
        valid_actions = game.getValidActions(cur_play, action)
        for i in range(40):
            self.search(copy.deepcopy(game),cur_play, action)
            # r = input(". Continue? [y|n]")
            # if r != "y":
            #     sys.exit()            

        s = game.stringRepresentation(states)

        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(game.getActionSize())]

        # print('couns {}'.format(counts))
        # 需修改成树搜索方式，本处暂时不要
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            # print('valid actions {} action prob --> {}'.format(counts, probs))
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        # print('temp -> {} counts -> {} probs -> {}'.format(temp, counts, probs))
        return probs



    def search(self,game, cur_play,action):
        # print('search play {} table {} action {}'.format(cur_play, game.getTableFrom(cur_play), action))
        # print('Qsa {}'.format(self.Qsa))
        # print('Nsa {}'.format(self.Nsa))
        # print('Es {}'.format(self.Es))
        # print('Ps {}'.format(self.Ps))
        # pdb.set_trace()
        
        states = game.getTableStates(cur_play,action)
        s = game.stringRepresentation(states)

        # ES  stores game.getGameEnded ended for board s
        if s not in self.Es:
            self.Es[s] = game.checkGameEnded(cur_play, action)

        if self.Es[s][cur_play] != 0:
            # terminal node
            return -self.Es[s][cur_play]

        # PS stores initial policy (returned by neural net)


        if s not in self.Ps:

            # leaf node
            self.Ps[s], v = self.nnet.predict(states)
            valids = game.getValidActions(cur_play, action)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                # 如果所有有效移动都被屏蔽，则使所有有效移动的可能性相等
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                # print("All valid moves were masked, do workaround.")
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

        # print('table {} states {} valids actions {}'.format(playerTable,s, valids))
        # pick the action with the highest upper confidence bound
        for a in range(game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    # u = v + (prob a * sqrt(s number / (1 + s,a number))
                    u = self.Qsa[(s,a)] + 1*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    # u = prob a * sqrt(s number + 0.001)
                    u = 1*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act

        next_player = game.getNextState(cur_play,a)
        # print('cur table {} next table {} next action {}'.format(playerTable, next_s, a))
        # next_s = self.game.getTableFrom(playerTable, next_player)

        v = self.search(game,next_player, a)

        if (s,a) in self.Qsa:
            # v(new) = (s,a number * v(old) + v(current)) / (s,a number + 1)
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)  # 计算平均V值 
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v




    


