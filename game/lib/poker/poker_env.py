import sys
sys.path.append('D:\\PROJECT_TW\\git\\myproject\\game')
from collections import deque
from lib.poker.play import Play
from lib.poker.deck import Deck
from lib.poker.card import Card
from lib.poker.action import Action
from lib.poker.poker_game import PokerGame 
from lib.poker.mcts import MCTS
from lib.poker.net import NNetWrapper as Net
from lib.poker.arena import Arena
import os
import time
import numpy as np
from pickle import Pickler, Unpickler
from random import shuffle

class PokerEnv(object):
    def __init__(self, num_plays=2):
        self.game = PokerGame(play_num=num_plays, card_num=12)     
        self.nnet = Net(state_num=8, action_num=self.game.getActionSize(), game=self.game)
        self.folder = 'D:\\PROJECT_TW\\git\\data\\game\\poker'
        self.nnet.load_checkpoint(self.folder,'best.pth.tar')
        self.pnet = self.nnet.__class__(state_num=8, action_num=self.game.getActionSize(),game=self.game)  # the competitor network
        self.mcts = MCTS(game=self.game, nnet=self.nnet, args=None)
        self.curPlayer = 0
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        self.args = None
        self.trainExamplesHistory = []
        

    def executeEpisode(self):
        trainExamples = []
        table = self.game.getInitTable()
        self.curPlayer = 0
        episodeStep = 0
        action = 0
        while(True):
            episodeStep +=1
            canonicalTable = self.game.getTableFrom(self.curPlayer)   
            temp = int(episodeStep < 5) 
            stateTable = self.game.getTableStates(canonicalTable)
            # print('ct --> {} st --> {}'.format(canonicalTable, stateTable))
            pi = self.mcts.getActionProb(stateTable,self.curPlayer, action, temp=temp) #
            # print('pi --> {}'.format(pi))
            action = np.random.choice(len(pi), p=pi)
            trainExamples.append([canonicalTable, self.curPlayer, pi, action])   #保存状态 
            canonicalTable, self.curPlayer = self.game.getNextState(self.curPlayer, action) 
            r = self.game.checkGameEnded(self.curPlayer, action)  # 返回得分
            if r[self.curPlayer] != 0:
                return [(x[0],r[x[1]],x[2],x[3]) for x in trainExamples]


    def learn(self, numIters):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1,numIters+1):
            
            if not self.skipFirstSelfPlay or i>1:
                # deque maxlen超过的将截掉
                iterationTrainExamples = deque([], maxlen=200000)
                for eps in range(1):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    iterationTrainExamples += self.executeEpisode()
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > 40:
                # print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)            

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            
            # shuffle examlpes before training
            # self.saveTrainExamples(i-1)
            # self.showTrainExample(self.trainExamplesHistory[-1])

            if i % 200 == 0:
                print('episodeStep {}'.format(i))
                self.showTrainExample(self.trainExamplesHistory[-1])

                trainExamples = []
                for e in self.trainExamplesHistory:
                    trainExamples.extend(e)
                # print('train examples -->{}  len -->{}'.format(trainExamples, len(trainExamples)))
                shuffle(trainExamples)
                
                # training new network, keeping a copy of the old one
                self.nnet.save_checkpoint(folder=self.folder, filename='temp.pth.tar')
                self.pnet.load_checkpoint(folder=self.folder, filename='temp.pth.tar')

                pmcts = MCTS(self.game, self.pnet, self.args)
                self.nnet.train(trainExamples)
                nmcts = MCTS(self.game, self.nnet, self.args)

                print('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(lambda x,y,z: np.argmax(pmcts.getActionProb(x, y, z, temp=0)),
                              lambda x,y,z: np.argmax(nmcts.getActionProb(x, y, z, temp=0)), self.game)
                pwins, nwins = arena.playGames(40)

                print('NEW/PREV WINS : %d / %d ;' % (nwins, pwins))
                if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < 0.6:
                    print('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.folder, filename='temp.pth.tar')
                else:
                    print('ACCEPTING NEW MODEL')
                    # self.nnet.save_checkpoint(folder=self.folder, filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.folder, filename='best.pth.tar')                

    def showTrainExample(self, example):
        for item in example:
            print('{} {} {}'.format(item[0], item[1], Card.from_id(item[3])))

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        # filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        filename = os.path.join(folder, 'best.pth.tar.examples')
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed        

    def loadTrainExamples(self):
        modelFile = os.path.join(self.folder, 'best.pth.tar')
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True        

if __name__ == '__main__':
    pv = PokerEnv()
    pv.loadTrainExamples()
    pv.learn(10000000)   
    
    
    

