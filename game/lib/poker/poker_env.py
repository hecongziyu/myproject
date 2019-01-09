from .play import Play

class Poker(object):
    def __init__(self, num_plays=2):
        self.num_plays = num_plays
        self.__init_plays__()
        
    
    def __init_plays__(self):
        self.plays = [Play(name='play_{}'.format(x)) for x in range(self.num_plays)]
        
    
    
    
    

