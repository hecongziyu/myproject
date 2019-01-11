from lib.poker.poker_env import Poker
from lib.poker.play import Play
from lib.poker.poker_brain import DQNet

EPOSE = 1
brain = DQNet()
poker = Poker()

def update(play):
    print('{}: {} action --> {}'.format(play.name,play.cards,play.chose_action(1)))
    



def run():
    for _ in range(EPOSE):
        poker.render()
        poker.start(2, update)    

if __name__ == '__main__':
    run()
