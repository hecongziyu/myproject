mcts state:
    Qsa : stores Q values for s,a (as defined in the paper)
    Nsa : stores #times edge s,a was visited
    Ns  : stores #times board s was visited
    Ps  : # stores initial policy (returned by neural net)
    Es  : # stores game.getGameEnded ended for board s
    Vs  : stores game.getValidMoves for board s


    state: private card, public card, last round action(???)































poker_env:

    self.mcts = MCTS(self.game, self.nnet, self.args)
    

    def executeEpisode():
        trainExamples = []
        table = self.game.getInitTable()        #初始化桌面, 随机分配
        self.curPlayer = 0

        episodeStep = 0
        while True:
            episodeStep +=1
            # 得到当前用户的桌面
            canonicalTable = self.game.getCanonicalFrom(table, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonicalTable, temp=temp) # 得到当前满足条件的action的概率
            sym = self.game.getSymmetries(canonicalTable, pi)  # 得到所有类似的board的概率, 处理流程是将boarder按90度翻转
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])   保存状态 
            action = np.random.choice(len(pi), p=pi)
            table, self.curPlayer = self.game.getNextState(table, self.curPlayer, action)            
            r = self.game.getGameEnded(table, self.curPlayer)  # 返回得分
            if r!=0:
                # 注意返回 (-1) ** (是否是当前用户)，如不为当前用户则为-1的相反
                # 返回格式 状态、价值、行动概率
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]    

    self.nnet = nnet
    self.pnet = self.nnet.__class__(self.game)  # the competitor network
    self.skipFirstSelfPlay = False 

    def learn():
        EPOCH = 100
        for i in range(EPOCH):
            # 重新初始化环境
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(iterationTrainExamples)            
            # 如果example history 超过



            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')


            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)


            print('两个网络进行比较')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                
          



poker_run:

    def run(play):
        observ = play.cards + poker_env.public_cards
        action = play.chose_action(observ, model, get_last_action())  # 根据当前状态选择行为
        reward = get_reward(play, observ, action)
        record(observ, action, reward)
        if number % 200 == 0:
            learn()
        

    def get_reward(play, observ, action):
        if len(play.cards) == 0:
            reward = 1
        reward = -1 ?????


play:

    def chose_action(observ, model, last_action):
        p = model.predict(observ)
        valid = self.get_valid_action()
        p = p * valid
        action = random(p)  从P中随机取一个值 ？？？
        return action
