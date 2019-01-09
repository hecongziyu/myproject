from .card import Card

class Play(object):
    def __init__(self, name="No Name"):
        self.name = name
#         self.uuid = uuid
        self.cards = []
#         self.stack = initial_stack
        self.action_histories = []
        # 检测player当前是否可以出牌
        self.is_activate = False
        
    def set_activate(self, activate):
        self.is_activate = activate
        
    def add_action_history(self):
        pass
    
    
    def set_cards(self, cards):
        self.cards = cards
        
    
    def chose_action(self, states):
        print('{} action --> {}'.format(name,'play'))
    
    
    