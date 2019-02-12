import sys
sys.path.append('D:\\PROJECT_TW\\git\\myproject\\game')

from lib.poker.play import Play
from lib.poker.deck import Deck
from lib.poker.card import Card
from lib.poker.state import States
import numpy as np

class PokerGame(object):
    def __init__(self, play_num, card_num=12):
        self.plays = []
        self.public_cards = []
        self.play_num = play_num
        self.card_num = card_num
        self.deck = Deck(cheat=True, cheat_card_ids=list(range(1,self.card_num+1)))

    def getActionSize(self):
        return self.card_num + 1

    def getInitTable(self):
        self.plays.clear()
        self.public_cards.clear()
        self.plays = [Play(uuid=x, name='play_{}'.format(x)) for x in range(self.play_num)]
        self.deck.shuffle()
        number = int(self.card_num/self.play_num)
        cards = np.split(np.array(self.deck.deck),self.play_num)
        for i in range(len(self.plays)):
            self.plays[i].set_cards(cards[i].tolist())
        return [self.plays[0].cards, self.public_cards]


    # 得到当前用户的桌面信息
    def getTableFrom(self, cur_play):
        return [self.plays[cur_play].cards.copy(),self.public_cards.copy()]

    def getTableStates(self, tables):
        
        # t = tables[0] + tables[1]
        s1 = [x.to_id() for x in tables[0]]  
        s1 = States.cards_to_states(s1)
        s2 = [x.to_id() for x in tables[1]]  
        s2 = States.cards_to_states(s2)
        s = s1 + s2
        # print(s1, s2, s)
        return s      


    # 得到下一个状态和用户
    def getNextState(self, cur_play, action):
        if action != 0:
            card = Card.from_id(action)
            if card in self.plays[cur_play].cards:
                self.plays[cur_play].remove_card(card)
                self.public_cards.append(card)
        cur_play = (cur_play+1)%self.play_num
        return [self.plays[cur_play].cards,self.public_cards], cur_play

    def getValidActions(self, player, action):
        valid = [0] * self.getActionSize()
        valid_num = 0
        # print('check valid action {} cards {}'.format(action, self.plays[player].cards))
        for item in self.plays[player].cards:
            if item.to_id() > action:
                valid[item.to_id()] = 1
                valid_num +=1
        if valid_num == 0:
            valid[0] = 1
        return valid

    # 检测是否结束，如已结束，返回相应状态
    # 参数 cur_play, action(上一个play的action)
    def checkGameEnded(self, cur_play, action):
        # print('check game end cur_play -> {} action -> {}'.format(cur_play,action))
        r = [0] * len(self.plays)
        # 检测是否有play的card为0
        for item in self.plays:
            if item.get_cards_num() == 0:

                return r

        # 当action为0时，当前用户只有一张时，该用户win，返回
        if action == 0 and self.plays[cur_play].get_cards_num() == 1:
            r = [x-1 for x in r]
            r[cur_play] = 1

            return r

        # 当action不为0时，当前用户只有一张时，检测该用户最后一张是否大于该action对应的card
        if action != 0 and self.plays[cur_play].get_cards_num() == 1:
            if self.plays[cur_play].cards[-1].to_id() > action:
                r = [x-1 for x in r]
                r[cur_play] = 1
                return r

        
        return r

    def stringRepresentation(self, states):
        # 8x8 numpy array (canonical board)
        return np.array(states).tostring()

if __name__ == '__main__':
    game = PokerGame(play_num=2, card_num=12)
    print(game.getInitTable())
    print(game.getValidActions(0,5))
