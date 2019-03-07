from lib.poker.card import Card


class Play(object):
    def __init__(self,uuid, name="No Name"):
        self.name = name
        self.uuid = uuid
        self.cards = []
        self.status = 'wait'

    
    def set_cards(self, cards):
        self.cards = cards
        self.cards.sort(key=lambda x: x.rank)
        self.status = 'playing'


    def remove_card(self, card):
        if card in self.cards:
            self.cards.remove(card)

    def get_cards_num(self):
        return len(self.cards)