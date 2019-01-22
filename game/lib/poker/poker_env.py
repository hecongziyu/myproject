from lib.poker.play import Play
from lib.poker.deck import Deck
from lib.poker.card import Card
from lib.poker.action import Action
from lib.poker.poker_game import PokerGame 
import time
import numpy as np

class PokerEnv(object):
    def __init__(self, num_plays=2):
        self.game = PokerGame(play_num=num_plays, card_num=12)     
        self.curPlayer = 0

    def executeEpisode(self):
        trainExamples = []
        table = self.game.getInitTable()
        self.curPlayer = 0
        episodeStep = 0

        while(True):
            episodeStep +=1
            canonicalTable = self.game.getTableFrom(table, self.curPlayer)   
            temp = int(episodeStep < 5) 
            pi = self.mcts.getActionProb(canonicalTable, temp=temp) #
            trainExamples.append([canonicalTable, self.curPlayer, pi, None])   保存状态     
            action = np.random.choice(len(pi), p=pi)
            self.curPlayer = self.game.getNextState(canonicalTable, self.curPlayer, action) 
            r = self.game.checkGameEnded(self.curPlayer, action)  # 返回得分
            
    
    
    

