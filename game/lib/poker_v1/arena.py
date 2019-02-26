import numpy as np
import copy
from lib.poker.card import Card

class Arena(object):
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            player 1 : old predict net
            player 2 : new trained predict net
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player1, self.player2]
        # 当前用户为1，使用player1来进行action选择，最后返回用户1（即play1）与play2对抗结果
        self.game.getInitTable()
        curPlayer = 0
        action = 0
        it = 0
        result = [0,0]

        while True:
            result = self.game.checkGameEnded(curPlayer, action)
            if result[0] != 0:
                break;
            it+=1
            # if verbose:
            #     assert(self.display)
            #     print("Turn ", str(it), "Player ", str(curPlayer))
            #     self.display(board)
            playerTable = self.game.getTableFrom(curPlayer) 
            # tableStates = self.game.getTableStates(playerTable)

            # 根据上一步的action 得到当前的用户的 action : t_action
            t_action = players[(curPlayer+1)%2](copy.deepcopy(self.game), curPlayer, action)  # ？？？？
            valids = self.game.getValidActions(curPlayer,action)
            if verbose:
                print('arena play {} table {} action {} '.format(curPlayer, playerTable, Card.from_id(t_action)))

            # valids[actions]==0 表非法action
            if valids[t_action]==0:
                print("not valid actions {}, valid --> {}, table state --> {}".format(action,valids, playerTable))

            # 确定该action是合法action
            # assert valids[action] >0
            curPlayer = self.game.getNextState(curPlayer, t_action)
            action = t_action

        # if verbose:
        #     assert(self.display)
        #     print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
        #     self.display(board)
        # 返回对手是否win的状态


        return result

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps = 0
        maxeps = int(num)
        num = int(num/2)
        oneWon = 0
        twoWon = 0
        for i in range(num):
            if i == num-1:
                print('arena old new game  play 0 --> trained new net ----------')
                gameResult = self.playGame(verbose=True)
                print('game result {}'.format(gameResult))
            else:
                gameResult = self.playGame(verbose=False)

            if gameResult[1]>0:
                oneWon+=1
            else:    # gameResult[1]==-1:
                twoWon+=1
            # bookkeeping + plot progress
            eps += 1

        self.player1, self.player2 = self.player2, self.player1
        
        for i in range(num):
            
            if i == num-1:
                print('arena switch new game  play 0 --> trained old net----------')
                gameResult = self.playGame(verbose=True)
                print('game result {}'.format(gameResult))
            else:
                gameResult = self.playGame(verbose=False)


            # print('game result --> {}'.format(gameResult))
            if gameResult[1]<0:
                oneWon+=1                
            else:  # gameResult[1]==1:
                twoWon+=1
            # bookkeeping + plot progress
            eps += 1

        return oneWon, twoWon
