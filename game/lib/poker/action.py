from lib.poker.card  import Card
import numpy as np
import math

class Action(object):

    def __init__(self):
        pass

    # 根据初始化的牌桌得到所有的有效的action
    # action map --> {id --> ["P",card_id]}
    @staticmethod
    def get_correct_actions(tables, card_num, play_num=2):
        # 单张的Action
        action_map_id = {}
        action_map_state = {}
        actions = np.zeros([play_num,int(card_num/play_num)],dtype=np.byte)
        for c in tables:
            actions[int(math.log2(c.suit))-1][c.rank-1] = 1
        action_value = np.sum(actions,axis=0)
        actions = []
        for i in range(0,2):
            flag = 'S' if i==0 else 'P'
            ats = (np.where(action_value>i)[0] ).tolist()
            action_map_id.update({x + (i*12+1):[flag,x+1] for x in ats})
            action_map_state.update({flag:[x+1 for x in ats]})
        return action_map_id, action_map_state


    # action = 0 是不要排
    @staticmethod
    def correct_action(play, action, action_history):
        correct = True
        card = ''
        last_action = 0
        if len(action_history) > 0:
            for item in list(reversed(action_history)):
                if item[1] > 0:
                    last_action = item[1]
                    break;

        if action > 0:
            card = Card.from_id(action)
            if card in play.cards:
                if len(action_history) > 0 and action < last_action:
                    correct = False
            else:
                correct = False
        else:
            if len(action_history) == 0:
                # 开始不能为0
                correct = False
            else:
                # 检测用户手中是否有大于action_history中最后一个大于0的
                for c in play.cards:
                    if Card.to_id(c) > last_action:
                        correct = False
                        break;


        # print('action check -->play cards:{} action:{} card:{} last action:{}'.format(play.cards,action, card, last_action))

        return correct

    # 选择该用户满足条件的动作列表
    @staticmethod
    def play_valid_actions(play_table, action_his_id, action_map):
        actions = []
        m_action_map , m_action_state= Action.get_correct_actions(play_table,len(action_map))

        if action_his_id != 0:
            in_action_state, in_action_value = action_map[action_his_id]
            m_action = [x for x in m_action_state[in_action_state] if x > in_action_value]
            if in_action_state == 'P':
                m_action = [x + 12 for x in m_action]
        else:
            m_action = []
            for k in m_action_state.keys():
                if k == 'P':
                    m_action += [x + 12 for x in m_action_state[k]]
                else:
                    m_action += [x for x in m_action_state[k]]


        return m_action


    @staticmethod
    def get_action_cards(action, action_map, card_lists):
        am = action_map[action]
        cards = [x for x in card_lists if x.rank==am[1]]
        if am[0] == 'S':
            # 单张
            cards = cards[0:1]
        return cards