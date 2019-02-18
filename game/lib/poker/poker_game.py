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

    # return [play_1_cards, play_2_cards, public_cards]
    def getInitTable(self):
        self.plays.clear()
        self.public_cards.clear()
        self.plays = [Play(uuid=x, name='play_{}'.format(x)) for x in range(self.play_num)]
        self.deck.shuffle()
        number = int(self.card_num/self.play_num)
        cards = np.split(np.array(self.deck.deck),self.play_num)
        for idx, x in enumerate(cards):
            self.plays[idx].set_cards(x.tolist())
        table = [self.plays[0].cards, self.plays[1].cards, self.public_cards]
        return table


    # 得到当前用户的桌面信息
    def getTableFrom(self, cur_play):
        return [self.plays[cur_play].cards, self.public_cards]

    # 得到当前用户的扑克信息，包括private card, public card, last action
    def getTableStates(self, cur_play,action):
        # print('get table states tables {} action {}'.format(tables,action))
        # t = tables[0] + tables[1]

        s1 = [x.to_id() for x in self.plays[cur_play].cards]  
        s1 = States.cards_to_states(s1)
        s2 = [x.to_id() for x in self.public_cards]  
        s2 = States.cards_to_states(s2)
        s = s1 + s2
        s.append(action)
        return s      


    def getTableStatesByTable(self, tables, action):
        s1 = [x.to_id() for x in tables[0]]
        s1 =  States.cards_to_states(s1)
        s2 = [x.to_id() for x in tables[1]]  
        s2 = States.cards_to_states(s2)
        s = s1 + s2
        s.append(action)
        return s      



    # 得到下一个状态和用户
    def getNextState(self, cur_play, action):
        if action != 0:
            card = Card.from_id(action)
            if card in self.plays[cur_play].cards:
                self.plays[cur_play].cards.remove(card)
                self.public_cards.append(card)
        cur_play = (cur_play+1)%self.play_num
        return cur_play

    def getValidActions(self,cur_play, action):
        valid = [0] * self.getActionSize()
        valid_num = 0
        # print('check valid action {} cards {}'.format(action, self.plays[player].cards))
        for item in self.plays[cur_play].cards:
            if item.to_id() > action:
                valid[item.to_id()] = 1
                valid_num +=1
        if valid_num == 0:
            valid[0] = 1
        # print('valid actions tables {} cur play {}  action {} valids {}'.format(tables,cur_play, action, valid))
        return valid

    # 检测是否结束，如已结束，返回相应状态
    # 参数 cur_play, action(上一个play的action)
    def checkGameEnded(self, cur_play, action):
        # print('check game end cur_play -> {} action -> {}'.format(cur_play,action))
        r = [0] * self.play_num
        # 检测是否有play的card为0
        for item in range(self.play_num):
            if self.plays[item].get_cards_num() == 0:
                r = [x-1 for x in r]
                r[item] = 1
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
        # print('string state {}'.format(states))
        # print('string {}'.format(np.array(states).reshape(-1)))
        return '|'.join('%s' % id for id in states)
        # s = ''
        # for item  in tables:
        #     for x in item:
        #         s = '{}{}'.format(s,x)
        # return s

if __name__ == '__main__':
    game = PokerGame(play_num=2, card_num=12)
    tables = game.getInitTable()
    print("init tables {}".format(tables))
    print("get play 0  tables{}".format(game.getTableFrom(0)))
    print("get play 1  tables{}".format(game.getTableFrom(1)))
    print('get play 0 valid actions {}'.format(game.getValidActions(0,0)))
    print('get play 1 valid actions {}'.format(game.getValidActions(1,0)))

    action = np.where(game.getValidActions(0,0))[0][0]
    game.getNextState(0,action)
    print('play 0 min action {} tables {}'.format(action, game.getTableFrom(0)))
    print("next states get play 1  tables{}".format(game.getTableFrom(1)))

    game.plays[1].cards.clear()
    print('check game ended {}'.format(game.checkGameEnded(0,0)))
    # print(game.getTableFrom(tables,1))
    # print(game.getValidActions(0,5))
