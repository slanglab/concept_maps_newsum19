import json

with open("publish/dataset_split.jsonl", "r") as inf:
    dt = json.load(inf)

print("[*] biggest")
print(max([len(_["scored_candidates"]) for _ in dt["training_set"]]))

class Instance(object):
    def __init__(self, raw, score, jdoc, mention_set_name):
        self.raw = raw
        self.score = score
        self.jdoc = jdoc
        self.mention_set_name = mention_set_name
        self.e1, self.e2 = self.mention_set_name.split("__")
        self.e1 = self.e1.replace("_", " ")
        self.e2 = self.e2.replace("_", " ")
        self.predicted = None


def get_tokens_before_tokens_in_and_tokens_after(instance):

    number_of_tokens_in_raw = len(instance.raw.lower().split())

    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    for i in find_ngrams(instance.jdoc["tokens"], number_of_tokens_in_raw):
        if [o["word"].lower() for o in i] == instance.raw.lower().split():
            return {"in": i}


def remove_e1_e2(instance, intoks):
    e1_e2 = instance.e1.lower().split() + instance.e2.lower().split()
    return [o for o in intoks["in"] if o["word"].lower() not in e1_e2]


def get_tokens_in_only(instance):
    return remove_e1_e2(instance, get_tokens_before_tokens_in_and_tokens_after(instance))


def deserialize(dataset, subset):
    for _ in dataset[subset]:
        for str_, score in _["scored_candidates"].items():
            jdoc = _['candidates_to_jdoc'][str_]
            yield Instance(raw=str_, score=score,
                           jdoc=jdoc, mention_set_name=_["name"])

train = [o for o in deserialize(dt, "training_set")]
test = [o for o in deserialize(dt, "test_set")]

trainN = len(set([p.mention_set_name for p in train]))
testN = len(set([p.mention_set_name for p in test]))


print("[*] len train = {}".format(trainN))
print("[*] len test = {}".format(testN))

all_ = list(train) + list(test)
len_toks = [len(get_tokens_in_only(_)) for _ in all_]
prob_toks = [_.jdoc["prob"] for _ in all_]
scores_toks = [_.score for _ in all_]

from scipy.stats import pearsonr
import numpy as np


print "[*] corr: length--scores"
print pearsonr(len_toks, scores_toks)



print "[*] corr: probs--scores"
print pearsonr(prob_toks, scores_toks)

print "[*] sanity check: assert all candidate probs > .5."

assert min([_.jdoc["prob"] for _ in all_]) > .5
