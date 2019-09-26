'''Modeling '''

import string
import numpy as np
import json
import mord

from sklearn import linear_model
from scipy.stats import spearmanr


def sigfigs(val, figs="3"):
    return str("{0:1." + figs + "f}").format(val)


class Instance(object):
    def __init__(self, raw, score, jdoc, mention_set_name):
        self.raw = raw
        self.score = score
        self.jdoc = jdoc
        self.mention_set_name = mention_set_name
        self.e1, self.e2 = self.mention_set_name.split("__")
        self.e1 = self.e1.replace("_", " ")
        self.e2 = self.e2.replace("_", " ")


def deserialize(dataset, subset):
    for _ in dataset[subset]:
        for str_, score in _["scored_candidates"].items():
            jdoc = _['candidates_to_jdoc'][str_]
            yield Instance(raw=str_, score=score,
                           jdoc=jdoc, mention_set_name=_["name"])


def set_to_x(set_):
    return np.concatenate([f.featurize(_) for _ in set_], axis=1).T


def set_to_y(set_):
    return np.asarray([_.score for _ in set_])


def fit_and_evaluate(training_set_, test_set_, model_):
    y_train_ = set_to_y(training_set_)
    X_train_ = set_to_x(training_set_)
    model_.fit(X_train_, y_train_)
    y_test_ = set_to_y(test_set_)
    X_test_ = set_to_x(test_set_)
    pred_test_ = model_.predict(X_test_)
    mae = metrics.mean_absolute_error(y_true=y_test_,
                                      y_pred=pred_test_)
    # first item in tuple is rho, second is pval
    spearman_model = spearmanr(pred_test_, y_test_)[0]
    return {'spearman': spearman_model,
            'mae': mae}


class Featurizer(object):

    def make_vectorizer(self, dataset, f):

        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(min_df=10)
        cv.fit([f(_) for _ in dataset])
        return cv

    def make_vectorizers(self, dataset):
        return {"endings_3": self.make_vectorizer(dataset, f=self.get_doc_as_morphemes),
                "pos": self.make_vectorizer(dataset, f=self.get_doc_as_pos),
                "word": self.make_vectorizer(dataset, f=self.get_doc_as_word),
                "lemma": self.make_vectorizer(dataset, f=self.get_doc_as_lemma)}

    def __init__(self, dataset):
        self.vectorizers = self.make_vectorizers(dataset)

    def name_to_entities(self, mention_set_name):
        e1, e2 = mention_set_name.split("__")
        return e1.replace("_", " "), e2.replace("_", " ")

    def strip_names(self, line):
        ln = line.raw.lower()
        e1, e2 = self.name_to_entities(line.raw)
        return ln.replace(e1, "").replace(e2, "").strip()

    def token_candidate(self, line):
        return " ".join([_["word"] for _ in self.get_tokens_before_tokens_in_and_tokens_after(line)["in"]])

    def f_is_one_tok(self, line):
        return len(self.remove_e1_e2(line, self.get_tokens_before_tokens_in_and_tokens_after(line))) <= 1

    def f_has_punct(self, line):
        for token in [_["word"] for _ in self.get_tokens_before_tokens_in_and_tokens_after(line)["in"]]:
            if token in string.punctuation:
                return True
        return False

    def get_tokens_before_tokens_in_and_tokens_after(self,instance):

        number_of_tokens_in_raw = len(instance.raw.lower().split())

        def find_ngrams(input_list, n):
            return zip(*[input_list[i:] for i in range(n)])

        for i in find_ngrams(instance.jdoc["tokens"], number_of_tokens_in_raw):
            if [o["word"].lower() for o in i] == instance.raw.lower().split():
                return {"in": i}

    def remove_e1_e2(self,instance,intoks):
        e1_e2 = instance.e1.lower().split() + instance.e2.lower().split()
        return [o for o in intoks["in"] if o["word"].lower() not in e1_e2]

    def get_tokens_in_only(self, instance):
        return self.remove_e1_e2(instance, self.get_tokens_before_tokens_in_and_tokens_after(instance))

    def f_has_said(self, line):
        return "said" in self.token_candidate(line)

    def f_p_is_statement(self, line):
        return line.jdoc['prob']

    def f_num_chars(self, line):
        return len(" ".join([_["word"] for _ in self.get_tokens_in_only(line)]))

    def f_starts_with_verb(self, line):
        a = self.remove_e1_e2(line, self.get_tokens_before_tokens_in_and_tokens_after(line))
        if len(a) > 0:
            return a[0]["pos"].lower()[0] == "v"
        else:
            return False

    def f_has_a_verb(self, line):
        a = self.remove_e1_e2(line,self. get_tokens_before_tokens_in_and_tokens_after(line))
        if len(a) > 0:
            is_v = any(i["pos"].lower()[0] == "v" for i in a)
        else:
            is_v = False
        return is_v

    def f_num_toks(self,line):
        return sum(1 for i in self.get_tokens_in_only(line))

    def get_doc_as_morphemes(self, instance):
        tokens = self.get_tokens_in_only(instance)
        return " ".join([_['word'][-2:] for _ in tokens])

    def get_doc_as_pos(self, instance):
        tokens = self.get_tokens_in_only(instance)
        return " ".join([_["pos"] for _ in tokens])

    def get_first_as_pos(self, instance):
        tokens = self.get_tokens_in_only(instance)
        if len(tokens) == 0:
            return ""
        else:
            return [_["pos"] for _ in tokens][0]

    def get_solo_as_pos(self, instance):
        tokens = self.get_tokens_in_only(instance)
        if len(tokens) == 1:
            return [_["pos"] for _ in tokens][0]
        else:
            return ""

    def get_doc_as_word(self, instance):
        tokens = self.get_tokens_in_only(instance)
        return " ".join([_["word"] for _ in tokens])

    def get_doc_as_lemma(self, instance):
        tokens = self.get_tokens_in_only(instance)
        return " ".join([_["lemma"] for _ in tokens])

    def featurize(self, line):

        standalone_feats = np.asarray([int(self.f_has_punct(line)),
                                       int(self.f_has_said(line)),
                                       int(self.f_num_toks(line)),
                                       int(self.f_num_chars(line)),
                                       int(self.f_starts_with_verb(line)),
                                       int(self.f_has_a_verb(line)),
                                       self.f_is_one_tok(line),
                                       float(self.f_p_is_statement(line))
                                       ]).reshape(-1, 1)
        endings_features = self.vectorizers["endings_3"].transform([self.get_doc_as_morphemes(line)]).todense().T
        pos_features = self.vectorizers["pos"].transform([self.get_doc_as_pos(line)]).todense().T
        word_features = self.vectorizers["lemma"].transform([self.get_doc_as_lemma(line)]).todense().T
        first_pos_features = self.vectorizers["pos"].transform([self.get_first_as_pos(line)]).todense().T
        solo_pos_features = self.vectorizers["pos"].transform([self.get_solo_as_pos(line)]).todense().T
        return np.concatenate([endings_features,
                               standalone_feats,
                               pos_features,
                               word_features,
                               first_pos_features,
                               solo_pos_features
                               ], axis=0).reshape(-1, 1)

    def label(self, line):
        return int(line["score"]) + 3 # plus three is needed for mord


if __name__ == "__main__":

    from sklearn import metrics

    with open("publish/dataset_split.jsonl", "r") as inf:
        dt = json.load(inf)

    TABLE = "output/results.tex"

    with open(TABLE, "w") as of:
        of.write("Model & Spearman's \\\\ \\midrule \n")

    training_set = list(deserialize(dataset=dt,
                                    subset='training_set'))

    test_set = list(deserialize(dataset=dt,
                                subset='test_set'))

    f = Featurizer(training_set)

    best = 0
    final_alpha = .01
    best_model = linear_model.LogisticRegression
    for alpha in [.001, .01, .1, 1, 10, 100, 1000]:
        for model in [mord.LogisticAT, mord.LogisticIT, mord.LogisticSE, linear_model.LogisticRegression]:

            spearmans = []
            from sklearn import cross_validation
            kf = cross_validation.KFold(len(training_set), n_folds=5)

            for train_index, test_index in kf:
                train_fold = [i for ino, i in enumerate(training_set)
                              if ino in train_index]
                test_fold = [i for ino, i in enumerate(training_set)
                             if ino in test_index]

                if model == linear_model.LogisticRegression:
                    clf2 = linear_model.LogisticRegression(
                                solver='lbfgs',
                                multi_class='multinomial',
                                C=alpha)
                else:
                    clf2 = model(alpha)

                spearman_fold = fit_and_evaluate(training_set_=train_fold,
                                                 test_set_=test_fold,
                                                 model_=clf2)["spearman"]

                spearmans.append(spearman_fold)

            print alpha, model, np.mean(spearmans)
            if np.mean(spearmans) > best:
                best = np.mean(spearmans)
                final_alpha = alpha
                best_model = model

    print "setting {} to final alpha".format(final_alpha)
    print "best model {}".format(best_model)

    clf2 = best_model(final_alpha)

    results = fit_and_evaluate(training_set_=training_set,
                               test_set_=test_set,
                               model_=clf2)

    with open(TABLE, "a") as of:
        spearman_model = sigfigs(results["spearman"], "3")
        of.write("{}&{} \\\\ \n".format("Model",spearman_model))

    clf2 = linear_model.LogisticRegression(
                solver='lbfgs',
                multi_class='multinomial',
                C=final_alpha)

    results = fit_and_evaluate(training_set_=training_set,
                               test_set_=test_set,
                               model_=clf2)

    spearman_lr = sigfigs(results["spearman"], "3")

    print "spearmanr LogisticRegression", spearman_lr
    with open(TABLE, "a") as of:
        of.write("{}&{}\\\\ \n".format("LogisticRegression", spearman_lr))
