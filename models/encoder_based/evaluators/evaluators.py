import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment
from evaluators.metrics import muc, b_cubed, ceafe, f1, phi4, lea

class MentionEvaluator: #compute mention f1 score
    def __init__(self):
        self.tp, self.fp, self.fn = 0, 0, 0

    def update(self, predicted_mentions, gold_mentions):
        predicted_mentions = set(predicted_mentions)
        gold_mentions = set(gold_mentions)

        self.tp += len(predicted_mentions & gold_mentions)
        self.fp += len(predicted_mentions - gold_mentions)
        self.fn += len(gold_mentions - predicted_mentions)

    def get_f1(self):
        pr = self.get_precision()
        rec = self.get_recall()
        return 2 * pr * rec / (pr + rec) if pr + rec > 0 else 0.0

    def get_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def get_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()
    
    def clear(self):
        self.tp, self.fp, self.fn = 0, 0, 0
        

class CorefEvaluator(object): #compute coref f1 score
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)
    
    def get_muc(self):
        return self.evaluators[0].get_precision(), self.evaluators[0].get_recall(), self.evaluators[0].get_f1()
    
    def get_b3(self):
        return self.evaluators[1].get_precision(), self.evaluators[1].get_recall(), self.evaluators[1].get_f1()
    
    def get_ceafe(self):
        return self.evaluators[2].get_precision(), self.evaluators[2].get_recall(), self.evaluators[2].get_f1()

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def clear(self):
        for e in self.evaluators:
            e.clear()
        

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den
    
    def clear(self):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0