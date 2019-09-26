'''
do the IAA analysis for this dataset
'''
from __future__ import division
import itertools
import random
import numpy as np
from scipy.stats import pearsonr
from code.display_set import load_mention_sets_and_display_sets

random.seed(42)

from tqdm import tqdm

def score_half_mention_set(display_sets_for_mention_set):
    '''
    Basically the same as the score_mention_set function, but there are no assert statements b/c there won't be 3 annotators.

    This is really only used for calculating IAA
    '''
    assert len(set([_.annotator for _ in display_sets_for_mention_set])) < 3  # i.e. fewer than 3 annos per set. This is cuz you split the annos in 1/2 for IAA 
    sentences_in_mention_set = set(itertools.chain(*[_.sentences for _ in display_sets_for_mention_set]))
    sent2score = {s:0 for s in sentences_in_mention_set}
    for display_set in display_sets_for_mention_set:
        for sentence in display_set.sentences:
            if sentence == display_set.best:
                sent2score[sentence] += 1
            if sentence == display_set.worst:
                sent2score[sentence] -= 1
    return sent2score  


def run_split_half_iaa():
    '''comptue IAA via split half reliability'''

    mention_sets_to_display_sets = load_mention_sets_and_display_sets()

    annotators = [1,2,3]

    total = []

    TRIALS = 1000

    for n in tqdm(range(TRIALS)):
        for mention_set, display_sets in mention_sets_to_display_sets.items():
            random.shuffle(annotators)

            first_half = score_half_mention_set([_ for _ in display_sets if _.annotator in annotators[:1]]).items()
            second_half = score_half_mention_set([_ for _ in display_sets if _.annotator in annotators[1:]]).items()

            first_half.sort(key=lambda x:x[0])  # sort by sentence, i.e. x[0] 
            second_half.sort(key=lambda x:x[0]) # sort by sentence, i.e. x[0]

            rank1 = [_[1] for _ in first_half]
            rank2 = [_[1] for _ in second_half]

            if len(rank1) == len(rank2) == 1:
                pass
            else:
                r, p = pearsonr(rank1, rank2)

                if not np.isnan(r): # happens if people disagree and ranks are all 0s
                    total.append(r) # e.g. rank1=[-1, 1], rank2=[0, 0] 

    print np.mean(total)


if __name__ == "__main__":
    print "[*] Split half IAA"
    run_split_half_iaa()
