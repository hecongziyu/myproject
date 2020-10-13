import pickle
import sys
import os
import codecs
import operator
from collections import defaultdict

from sentence import Sentence
from config import Config

class BREDS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, sentences_file, similarity, confidence):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds,sentences_file,similarity, confidence)

    
    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences
        where named entities are already tagged
        """
        try:
            os.path.isfile("processed_tuples.pkl")
            f = open("processed_tuples.pkl", "rb")
            print("\nLoading processed tuples from disk...")
            self.processed_tuples = pickle.load(f)
            f.close()
            print(len(self.processed_tuples), "tuples loaded")

        except IOError:
            print("\nGenerating relationship instances from sentences")
            f_sentences = codecs.open(sentences_file, encoding='utf-8')
            count = 0
            for line in f_sentences:
                count += 1
                if count % 10000 == 0:
                    sys.stdout.write(".")
                sentence = Sentence(line.strip(), 
                                    self.config.e1_type, 
                                    self.config.e2_type, 
                                    self.config.max_tokens_away,
                                    self.config.min_tokens_away, 
                                    self.config.context_window_size)

                for rel in sentence.relationships:
                    if rel.arg1type == self.config.e1_type and rel.arg2type == self.config.e2_type:
                        bef_tokens = word_tokenize(rel.before)
                        bet_tokens = word_tokenize(rel.between)
                        aft_tokens = word_tokenize(rel.after)
                        if not (bef_tokens == 0 and bet_tokens == 0 and aft_tokens == 0):
                            # print('rel :', rel.ent1, ',', rel.ent2, ',', rel.sentence, ',', rel.before)
                            t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, 
                                      self.config)
                            self.processed_tuples.append(t)
            f_sentences.close()

            print("\n", len(self.processed_tuples), "relationships generated")
            print("Dumping relationships to file")
            # f = open("processed_tuples.pkl", "wb")
            # pickle.dump(self.processed_tuples, f)
            # f.close()

    def init_bootstrapp(self, tuples):
        if tuples is not None:
            f = open(tuples, "rb")
            print("Loading pre-processed sentences", tuples)
            self.processed_tuples = pickle.load(f)
            f.close()
            print(len(self.processed_tuples), "tuples loaded")

        """
        starts a bootstrap iteration
        """
        i = 0
        while i <= self.config.number_iterations:
            print("\n=============================================")
            print("\nStarting iteration", i)
            print("\nLooking for seed matches of:")
            for s in self.config.seed_tuples:
                print(s.e1, '\t', s.e2)


        






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
    similarity = 0.5
    confidance = 0.5


    print('begin ....')
    breads = BREDS(join(data_root,configuration), 
                   join(data_root,seeds_file), 
                   join(data_root,negative_seeds), 
                   join(data_root, sentences_file),
                   similarity, 
                   confidance)


        # if sentences_file.endswith('.pkl'):
        #     print("Loading pre-processed sentences", sentences_file)
        #     breads.init_bootstrap(tuples=sentences_file)
        # else:
    breads.generate_tuples(join(data_root,sentences_file))

    print('end ...')
    # breads.init_bootstrap(tuples=None)
