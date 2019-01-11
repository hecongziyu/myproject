from lib.poker.play import Play
from lib.poker.deck import Deck
import time

class Poker(object):
    def __init__(self, num_plays=2):
        self.num_plays = num_plays
        self.play_actions_his = []
        self.public_cards = []
        self.__init_plays__()


        
    
    def __init_plays__(self):
        self.plays = [Play(name='play_{}'.format(x)) for x in range(self.num_plays)]
        
    def play_reset(self):
        for item in self.plays:
            item.reset()

    def poker_play(self, play, action):

        return self.public_cards


    # 刷新        
    def render(self):
        self.play_actions_his.clear()
        self.public_cards.clear()
        self.play_reset()
        deck = Deck(cheat=True, cheat_card_ids=list(range(1,13)))
        deck.shuffle()
        number = int(len(deck.deck)/len(self.plays))
        self.plays[0].set_cards(deck.deck[0:number])
        self.plays[1].set_cards(deck.deck[number:])



    def start(self,sleep_time, update):
        while True:
            for play in self.plays:
                time.sleep(sleep_time)
                status = update(play)
                if status == 'done':
                    break

            if status == 'done':
                print('game exist')
                break







        
#         print(self.plays[0].cards)
#         for item in deck.deck:
#             print(item)
        
        
        
if __name__ == '__main__':
    poker = Poker()
    print(poker.plays)
        
    
    
    
    

