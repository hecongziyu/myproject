class Card(object):
    CLUB = 2
    DIAMOND = 4
    HEART = 8
    SPADE = 16

    SUIT_MAP = {
        2  : 'C',
        4  : 'D',
        8  : 'H',
        16 : 'S'}

    RANK_MAP = {
	    12  :  'A',
      13  :  '2',
      1  :  '3',
      2  :  '4',
      3  :  '5',
      4  :  '6',
      5  :  '7',
      6  :  '8',
      7  :  '9',
      8 : 'T',
      9 : 'J',
      10 : 'Q',
      11 : 'K'}


    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __repr__(self):
        suit = self.SUIT_MAP[self.suit]
        rank = self.RANK_MAP[self.rank]
        return '{}{}'.format(suit, rank)

    def to_id(self):
        rank = self.rank
        num = 0
        tmp = self.suit >> 1
        while tmp&1 != 1:
            num += 1
            tmp >>= 1

        return rank + 12 * num

    @classmethod
    def from_id(cls, card_id):
        if card_id >0:
          suit, rank = 2, card_id
          while rank > 12:
              suit <<= 1
              rank -= 12
          return cls(suit, rank)
        else:
          return 0

    @classmethod
    def from_str(cls, str_card):
        assert(len(str_card)==2)
        inverse = lambda hsh: {v:k for k,v in hsh.items()}
        suit = inverse(cls.SUIT_MAP)[str_card[0].upper()]
        rank = inverse(cls.RANK_MAP)[str_card[1]]
        return cls(suit, rank)
    
