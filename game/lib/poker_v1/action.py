from lib.poker.card  import Card

class Action(object):

    def __init__(self):
        pass

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
    def play_valid_actions(play, action_history):
        last_action = 0 if len(action_history) == 0 else action_history[-1][-1]
        actions = [Card.to_id(x) for x in play.cards if Card.to_id(x)  > last_action]
        return actions