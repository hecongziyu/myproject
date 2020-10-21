import pickle
import sys
import os
import codecs
import operator
from collections import defaultdict
from nltk import word_tokenize
from gensim.matutils import cossim

from .sentence import Sentence
from .config import Config
from .pattern import Pattern
from .tuples import Tuple
from .seed import Seed

PRINT_PATTERNS = True

class BREDS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, sentences_file, similarity, confidence):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        print('config file ---->', config_file)
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

                        # print('rel between :', rel.between)
                        bef_tokens = word_tokenize(rel.before)
                        bet_tokens = word_tokenize(rel.between)
                        aft_tokens = word_tokenize(rel.after)
                        # print('rel between token :', bet_tokens)
                        if not (bef_tokens == 0 and bet_tokens == 0 and aft_tokens == 0):
                            # print('rel :', rel.ent1, ',', rel.ent2, ',', rel.sentence, ',', rel.before)
                            t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, 
                                      self.config)
                            self.processed_tuples.append(t)
            f_sentences.close()

            print("\n", len(self.processed_tuples), "relationships generated")
            print("Dumping relationships to file")
            f = open("processed_tuples.pkl", "wb")
            pickle.dump(self.processed_tuples, f)
            f.close()

    def init_bootstrap(self, tuples):

        print('init boot strap start .....')

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

            count_matches, matched_tuples = self.match_seeds_tuples(self)

            print('matched tuples :', matched_tuples)
            if len(matched_tuples) == 0:
                print("\nNo seed matches found")
                sys.exit(0)

            print("\nNumber of seed matches found %s" % count_matches)     
            sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True) 

            # for t in sorted_counts:
            #     print(t[0][0], '\t', t[0][1], t[1])            

            print("\nClustering matched instances to generate patterns")
            self.cluster_tuples(self, matched_tuples)



            # Eliminate patterns supported by less than 'min_pattern_support' tuples

            new_patterns = [p for p in self.patterns if len(p.tuples) >= self.config.min_pattern_support]
            self.patterns = new_patterns
          
            for p in self.patterns:
                print('len tuples:', len(p.tuples), 'sentence :', '\n'.join([x.sentence for x in p.tuples]))

            print("\n", len(self.patterns), "patterns generated")
            if i == 0 and len(self.patterns) == 0:
                print("No patterns generated")
                sys.exit(0)

            # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
            # This was already collect and its stored in: self.processed_tuples
            #
            # Measure the similarity of each occurrence with each extraction pattern
            # and store each pattern that has a similarity higher than a given threshold
            #
            # Each candidate tuple will then have a number of patterns that helped generate it,
            # each with an associated de gree of match. Snowball uses this infor
            print("\nCollecting instances based on extraction patterns")
            count = 0
            pattern_best = None
            for t in self.processed_tuples:
                count += 1
                if count % 100 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                sim_best = 0

                for extraction_pattern in self.patterns:
                    score = self.similarity(t, extraction_pattern)
                    if score > self.config.threshold_similarity:
                        extraction_pattern.update_selectivity(t, self.config)
                    if score > sim_best:
                        sim_best = score
                        pattern_best = extraction_pattern

                if sim_best >= self.config.threshold_similarity:
                    # if this instance was already extracted, check if it was by this extraction pattern
                    # print('tubles -->', t.bef_words)
                    # print('ca tuples -->', t, ' type:', type(t))
                    patterns = self.candidate_tuples[t]
                    if patterns is not None:
                        if pattern_best not in [x[0] for x in patterns]:
                            self.candidate_tuples[t].append((pattern_best, sim_best))
                    # if this instance was not extracted before, associate theisextraciton pattern with the instance
                    # and the similarity score
                    else:
                        self.candidate_tuples[t].append((pattern_best, sim_best))


                    extraction_pattern.confidence_old = extraction_pattern.confidence
                    extraction_pattern.update_confidence()

            # normalize patterns confidence
            # find the maximum value of confidence and divide all by the maximum
            max_confidence = 0
            for p in self.patterns:
                if p.confidence > max_confidence:
                    max_confidence = p.confidence
                if max_confidence > 0:
                    for p in self.patterns:
                        p.confidence = float(p.confidence) / float(max_confidence)

            if PRINT_PATTERNS is True:
                print("\nPatterns:")
                for p in self.patterns:
                    p.merge_tuple_patterns()
                    print("Patterns:", len(p.tuples), ' sentence:', '|||'.join([x.sentence for x in p.tuples]))
                    print("Positive", p.positive)
                    print("Negative", p.negative)
                    print("Unknown", p.unknown)
                    print("Tuples", len(p.tuples))
                    print("Pattern Confidence", p.confidence)
                    print("\n")

            # update tuple confidence based on patterns confidence
            print("\nCalculating tuples confidence")
            for t in self.candidate_tuples.keys():
                confidence = 1
                t.confidence_old = t.confidence
                for p in self.candidate_tuples.get(t):
                    confidence *= 1 - (p[0].confidence * p[1])

                t.confidence = 1 - confidence

                if i > 0:
                    t.confidence = t.confidence * self.config.wUpdt + t.confidence_old * (1 - self.config.wUpdt)

                # update seed set of tuples to use in next iteration
                # seeds = { T | Conf(T) > min_tuple_confidence }

            if i+1 < self.config.number_iterations:
                print("Adding tuples to seed with confidence =>" + str(self.config.instance_confidance))
                for t in self.candidate_tuples.keys():
                    if t.confidence >= self.config.instance_confidance:
                        seed = Seed(t.e1, t.e2)
                        self.config.seed_tuples.add(seed)

                # increment the number of iterations


            i += 1


        print("\nWriting extracted relationships to disk")
        f_output = open("relationships.txt", "w", encoding='utf-8')
        tmp = sorted(self.candidate_tuples, key=lambda tpl: tpl.confidence, reverse=True)
        for t in tmp:
            f_output.write("instance: "+t.e1+'\t'+t.e2+'\tscore:'+str(t.confidence)+'\n')
            f_output.write("sentence: "+t.sentence+'\n')
            # writer patterns that extracted this tuple
            patterns = set()
            for pattern in self.candidate_tuples[t]:
                patterns.add(pattern[0])
            for p in patterns:
                p.merge_tuple_patterns()
                f_output.write("pattern_bet: " + ', '.join(p.tuple_patterns) + '\n')
            if t.passive_voice is False or t.passive_voice is None:
                f_output.write("passive voice: False\n")
            elif t.passive_voice is True:
                f_output.write("passive voice: True\n")
            f_output.write("\n")
        f_output.close()
            

    @staticmethod
    def cluster_tuples(self, matched_tuples):
        """
        single-pass clustering
        """
        start = 0
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)
            start = 1

        # Compute the similarity between an instance with each pattern go through all tuples
        for i in range(start, len(matched_tuples), 1):
            t = matched_tuples[i]
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the
            # highest similarity score
            for w in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[w]
                score = self.similarity(t, extraction_pattern)
                # print('t bet vector:', t.bet_vector, ' pattern vetor:', extraction_pattern.centroid_bet)
                # print('t sentence :', t.sentence, ' score :', score, ' patterns sentence:', '||'.join([x.sentence for x in extraction_pattern.tuples]))

                if score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = w

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)



        

    @staticmethod
    def match_seeds_tuples(self):
        """
        checks if an extracted tuple matches seeds tuples
        """
        matched_tuples = list()
        count_matches = dict()
        for t in self.processed_tuples:
            for s in self.config.seed_tuples:
                if t.e1 == s.e1 and t.e2 == s.e2:
                    # t Tuple _hash_ 包括(bef,e1,bet,e2,aft)
                    matched_tuples.append(t)
                    try:
                        count_matches[(t.e1, t.e2)] += 1
                    except KeyError:
                        count_matches[(t.e1, t.e2)] = 1

        return count_matches, matched_tuples


    def similarity(self, t, extraction_pattern):
        (bef, bet, aft) = (0, 0, 0)
        if t.bef_vector is not None and extraction_pattern.centroid_bef is not None:
            bef = cossim(t.bef_vector, extraction_pattern.centroid_bef)
        if t.bet_vector is not None and extraction_pattern.centroid_bet is not None:
            bet = cossim(t.bet_vector, extraction_pattern.centroid_bet)
        if t.aft_vector is not None and extraction_pattern.centroid_aft is not None:
            aft = cossim(t.aft_vector, extraction_pattern.centroid_aft)
        return self.config.alpha*bef + self.config.beta*bet + self.config.gamma*aft




