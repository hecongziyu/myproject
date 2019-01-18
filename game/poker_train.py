from lib.poker.poker_env import Poker
from lib.poker.play import Play
from lib.poker.poker_brain import DQNet
from lib.poker.state import States


EPOSE = 10000000
update_count = 1

def update(play):
    global update_count


    pstate = States.cards_to_states(play.get_play_cards_id()) +  States.cards_to_states(poker_env.get_public_cards_id())
    # pstate = [round(x/255,2) for x in pstate]
    action = qnet.choss_action(pstate)
    state,reward,done = poker_env.poker_play(play, action)
    pstate_ = States.cards_to_states(play.get_play_cards_id()) +  States.cards_to_states(poker_env.get_public_cards_id())
    # pstate_ = [round(x/255,2) for x in pstate_]
    
    qnet.store_transition(pstate,action,reward,pstate_)
    if update_count % 100 == 0:
        print('states :{} states_ :{} action :{} reard :{}'.format(pstate, pstate_, action, reward))
        qnet.learn()
        update_count = 1
    update_count += 1

    # print('{}: {} :{} : {} : {}'.format(play.name,action,states,reward))
    return done


def run():
    for _ in range(EPOSE):
        poker_env.render()
        poker_env.start(0.001, update)    

if __name__ == '__main__':
    qnet = DQNet()
    poker_env = Poker()
    run()
