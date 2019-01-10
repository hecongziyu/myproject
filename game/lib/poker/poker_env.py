from lib.poker.play import Play
from lib.poker.deck import Deck

class Poker(object):
    def __init__(self, num_plays=2):
        self.num_plays = num_plays
        self.__init_plays__()
        
    
    def __init_plays__(self):
        self.plays = [Play(name='play_{}'.format(x)) for x in range(self.num_plays)]
        
    def reset(self):
        for item in self.plays:
            item.reset()
            
    def begin(self):
        deck = Deck(cheat=True, cheat_card_ids=list(range(1,13)))
        deck.shuffle()
        number = int(len(deck.deck)/len(self.plays))
        
        self.plays[0].set_cards(deck.deck[0:number])
        self.plays[1].set_cards(deck.deck[number:])
        
#         print(self.plays[0].cards)
#         for item in deck.deck:
#             print(item)
        
        
        
if __name__ == '__main__':
    poker = Poker()
    print(poker.plays)
        
    
    
    
    

