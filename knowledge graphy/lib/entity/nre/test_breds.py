from os.path import join
from breds.breds import BREDS
import pickle


def test_breds(breds):



    breds.init_bootstrap(tuples='processed_tuples.pkl')
    

def test_breds_cluster_tuples(breds):

    f = open('processed_tuples.pkl', "rb")
    print("Loading pre-processed sentences")
    breds.processed_tuples = pickle.load(f)
    f.close()
    print(len(breds.processed_tuples), "tuples loaded")

    count_matches, matched_tuples = breds.match_seeds_tuples(breds)
    print('count matches ', count_matches, ' matched_tuples:', matched_tuples)
    
    # for titem in matched_tuples:
        # print(titem.sentence, ' bet vector:', titem.bet_vector, ' bet words:', titem.bet_words, ' e1:', titem.e1, ' e2:', titem.e2)

    print('before patterns :', breds.patterns)
    breds.cluster_tuples(breds, matched_tuples)
    print('after patterns :', breds.patterns)

    for idx, p in enumerate(breds.patterns):
        print('idx :', idx, ' breds patterns sentences:', '|||'.join([x.sentence for x in p.tuples]))






if __name__ == '__main__':
    # if len(sys.argv) != 7:
    #     print("\nBREDS.py parameters sentences positive_seeds negative_seeds "
    #           "similarity confidence\n")
    #     sys.exit(0)
    # else:
    from os.path import join
    data_root = r'D:\PROJECT_TW\git\data\kg\entity\nre'
    configuration = 'parameters.cfg'
    sentences_file = 'sentence_file.txt'
    seeds_file = 'seeds_positive.txt'
    negative_seeds = 'seeds_negative.txt'
    similarity = 0.4
    confidance = 0.5


    print('begin ....')


        # if sentences_file.endswith('.pkl'):
        #     print("Loading pre-processed sentences", sentences_file)
        #     breads.init_bootstrap(tuples=sentences_file)
        # else:
    # breads.generate_tuples(join(data_root,sentences_file))
    # breads.init_bootstrap(tuples=None)

    breds = BREDS(configuration, 
                   join(data_root,seeds_file), 
                   join(data_root,negative_seeds), 
                   join(data_root, sentences_file),
                   similarity, 
                   confidance)   
    # breds.generate_tuples(join(data_root,sentences_file)) 
    test_breds(breds)

    # test_breds_cluster_tuples(breds)

    print('end ...')
    
