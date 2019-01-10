from lib.poker.poker_env import Poker

if __name__ == '__main__':
    poker = Poker()
    poker.begin()
    
    for item in poker.plays:
        print(item.name)   
