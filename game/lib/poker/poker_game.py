import sys
sys.path.append('D:\\PROJECT_TW\\git\\myproject\\game')

from lib.poker.play import Play
from lib.poker.deck import Deck
from lib.poker.card import Card
from lib.poker.state import States
from lib.poker.action import Action
import numpy as np

class PokerGame(object):
    def __init__(self, play_num, card_num=12):
        self.plays = []
        self.public_cards = []
        self.play_num = play_num
        self.card_num = card_num
        self.deck = Deck(cheat=True, cheat_card_ids=list(range(1,self.card_num+1)))
        self.action_map_id, self.action_map_state = Action.get_correct_actions(self.deck.deck, self.card_num, self.play_num)

    

    def getActionSize(self):
        return len(self.action_map_id) + 1

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
        tables = [self.plays[cur_play].cards, self.public_cards]
        return  self.getTableStatesByTable(tables, action)


    def getTableStatesByTable(self, tables, action):
        s1 = [x.to_id() for x in tables[0]]
        s1 =  States.cards_to_states(s1, states_number=24)
        s2 = [x.to_id() for x in tables[1]]  
        s2 = States.cards_to_states(s2, states_number=24)
        s = s1 + s2
        s.append(action)
        return s      

    # 根据card rank比较
    def __remove_cards_from_self__(self,cur_play, card):
        remove_card = None
        for c in self.plays[cur_play].cards:
            if c.rank == card.rank and c.suit == card.suit:
                remove_card = c
                self.public_cards.append(c)
                break
        if remove_card is not None:
            self.plays[cur_play].cards.remove(remove_card)

    # 得到下一个状态和用户
    def getNextState(self, cur_play, action):
        if action != 0:
            cards = Action.get_action_cards(action,self.action_map_id,self.plays[cur_play].cards)
            # print('next state cur_play {} action {} cards {}'.format(cur_play, action, cards))
            for c in cards:
                self.__remove_cards_from_self__(cur_play,c)
        cur_play = (cur_play+1)%self.play_num
        return cur_play

    def getValidActions(self,cur_play, action):
        valids = np.zeros(self.getActionSize(),dtype=np.byte)
        # if action != 0:
        pv = Action.play_valid_actions(self.plays[cur_play].cards, 
            action, 
            self.action_map_id)
        valids[pv] = 1
        if np.sum(valids) == 0:
            valids[0] = 1
        return valids
        

    # 检测是否结束，如已结束，返回相应状态
    # 参数 cur_play, action(上一个play的action)
    def checkGameEnded(self, cur_play, action):
        # print('check game end cur_play -> {} action -> {}'.format(cur_play,action))
        r = [0] * self.play_num
        # 对方剩余card数
        
        

        # 检测是否有play的card为0
        for item in range(self.play_num):
            if self.plays[item].get_cards_num() == 0:
                r = [x-1 for x in r]
                r[item] = 1
                break;

        # 当action为0时，当前用户只有一张时，该用户win，返回
        if r[0]==0 and action == 0 and self.plays[cur_play].get_cards_num() == 1:
            r = [x-1 for x in r]
            r[cur_play] = 1

        # 当action不为0时，当前用户只有一张时，检测该用户最后一张是否大于该action对应的card
        if r[0]==0 and  action != 0 and self.plays[cur_play].get_cards_num() == 1:
            if self.plays[cur_play].cards[-1].to_id() > action:
                r = [x-1 for x in r]
                r[cur_play] = 1
        
        if r[0] != 0:
            ridx = r.index(-1)
            nc = round((self.plays[ridx].get_cards_num()) / int(self.card_num/self.play_num),3)
            # print('c play card number {} {}'.format(self.plays[ridx].get_cards_num(),nc))
            r = [nc] * self.play_num
            r[ridx] = -nc

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
    # Set-ExecutionPolicy RemoteSigned
    game = PokerGame(play_num=2, card_num=24)
    tables = game.getInitTable()
    tablesIds = [[x.to_id() for x in tb] for tb in tables]
    print("init tables {}".format(tables))
    print('init {}'.format(game.action_map_state))
    print('init {}'.format(game.action_map_id))
    # print('deck {}'.format(game.deck.deck))
    # print('action {}'.format(Action.get_correct_actions(game.deck.deck,24,2)))
    print('play 0 valid action {}'.format(game.getValidActions(0,0)))
    # print(game.deck.deck)
    # game.getNextState(0,3)
    # print("get play 0  tables{}".format(game.getTableFrom(0)))
    # print("get play 1  tables{}".format(game.getTableFrom(1)))
    # print('get play 0 valid actions {}'.format(game.getValidActions(0,0)))
    # print('get play 1 valid actions {}'.format(game.getValidActions(1,0)))

    # action = np.where(game.getValidActions(0,13))[0][0]
    # print(action)
    # game.getNextState(0,action)
    # print('play 0 min action {} tables {}'.format(action, game.getTableFrom(0)))
    print('---> {}'.format(game.getTableStates(0,1)))
    # print("next states get play 1  tables{}".format(game.getTableFrom(1)))

    # game.plays[0].cards.clear()
    # print('check game ended {}'.format(game.checkGameEnded(1,0)))
    # print(game.getTableFrom(tables,1))
    # print(game.getValidActions(0,5))
