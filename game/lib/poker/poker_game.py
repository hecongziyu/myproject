import sys
sys.path.append('D:\\PROJECT_TW\\git\\myproject\\game')

from lib.poker.play import Play
from lib.poker.deck import Deck
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
        return [self.plays[cur_play],self.public_cards]

    # 得到下一个状态和用户
    def getNextState(self, table, cur_play, action):
        if action != 0:
            card = Cards.from_id(action)
            self.plays[cur_play].remove_card(card)
            self.public_cards.append[card]
        cur_play = (cur_play+1)%self.play_num
        return [self.plays[cur_play],self.public_cards]

    # 检测是否结束，如已结束，返回相应状态
    # 参数 cur_play, action(上一个play的action)
    def checkGameEnded(self, cur_play, action):
        r = [(int(self.card_num/self.play_num) - x.get_cards_num())/int(self.card_num/self.play_num) for x in self.plays]
        # 检测是否有play的card为0
        for item in self.plays:
            if item.get_cards_num() == 0:
                return r

        # 当action为0时，当前用户只有一张时，该用户win，返回
        if action == 0 && self.plays[cur_play].get_cards_num() == 1:
            r[self.cur_play] = 1
            return r

        # 当action不为0时，当前用户只有一张时，检测该用户最后一张是否大于该action对应的card
        if action != 0 && self.plays[cur_play].get_cards_num() == 1:
            if self.plays[cur_play].cards[-1] > action:
                r[self.cur_play] = 1
                return r

        r = [0] * len(self.plays)
        return r

if __name__ == '__main__':
    game = PokerGame(play_num=2, card_num=12)
    game.getInitTable()
    for a in game.plays:
        print(a.cards)
