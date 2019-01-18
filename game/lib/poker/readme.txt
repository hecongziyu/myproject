poker_env:

    def executeEpisode():
        trainExamples = []
        table = self.game.getInitTable()        #初始化桌面
        self.curPlayer = 0


        while True:
            
            canonicalTable = self.game.getCanonicalFrom(table, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonicalTable, temp=temp) # 得到当前满足条件的action的概率
            sym = self.game.getSymmetries(canonicalTable, pi)  # 得到桌面每个点可能的概率
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])   保存状态 
            action = np.random.choice(len(pi), p=pi)
            table, self.curPlayer = self.game.getNextState(table, self.curPlayer, action)            
            r = self.game.getGameEnded(table, self.curPlayer)  # 返回得分
            if r!=0:
                # 注意返回 (-1) ** (是否是当前用户)，如不为当前用户则为-1的相反
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]       

          



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
