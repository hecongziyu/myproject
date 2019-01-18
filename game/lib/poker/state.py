# https://github.com/ilanschnell/bitarray
from bitarray import bitarray
import numpy as np

class States(object):


    @staticmethod
    def cards_to_states(cards, states_number=16, split_number=1):
        cardmap = np.array([True]*states_number)
        cardmap[cards] = False
        cardmap = np.hsplit(cardmap,4)
        states = [bitarray(x.tolist()) for x in cardmap]
        
        states = [int.from_bytes(x.tobytes(),byteorder='big') for x in states]
        return states
        


    @staticmethod
    def states_to_cards(states, states_number=14, split_number=1):
        assert len(states) == 4
        # bitarray 返回frombytes是返回的8 bit，reserve=4是只取高4位
        def int_to_bitarray(x, reserve=4):
            b = bitarray()
            b.frombytes(x.to_bytes(1,byteorder='big'))
            return b.tolist()[0:4]
        statemap = [int_to_bitarray(x) for x in states]
        statemap = np.array(statemap).reshape(-1)
        cards = np.where(statemap==False)[0].tolist()
        print(statemap)
        print(cards)

if __name__ == '__main__':
    state = States.cards_to_states([0,1,3,4,7,8,9,11,13])
    print(state)
    States.states_to_cards(state)
