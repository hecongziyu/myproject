from lib.poker.card import Card


class Play(object):
    def __init__(self, name="No Name"):
        self.name = name
#         self.uuid = uuid
        self.cards = []
#         self.stack = initial_stack
        self.action_histories = []
        # 检测player当前是否可以出牌
        self.is_activate = False
        # status : win , loss, wait, playing
        self.status = 'wait'
        
    def set_activate(self, activate):
        self.is_activate = activate
        
    def add_action_history(self):
        self.action_histories.append(action)
    
    def set_cards(self, cards):
        self.cards = cards
        self.status = 'playing'

    def play_card(self, card):
        if card in self.cards:
            self.cards.remove(card)
            self.action_histories.append(card)

    def get_play_cards_id(self):
        return [Card.to_id(x) for x in self.cards]

        
    def reset(self):
        self.cards.clear()
        self.action_histories.clear()
        self.is_activate = False
        self.status = 'wait'

        
    
    
    
    