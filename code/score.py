import time
import pickle
from tqdm import tqdm
from pyswip import Prolog
from wrapt_timeout_decorator import *

def get_stats(tp:int, fp:int, fn:int):
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    return precision, recall, f1

def score_sets(pred:set, gt:set):
    true_pos_count = len(pred&gt)
    false_pos_count = len(pred)-true_pos_count
    false_neg_count = len(gt) - true_pos_count
    return true_pos_count, false_pos_count, false_neg_count

def print_scores(results):
    tp = 0
    fp = 0
    fn = 0
    for k,v in results.items():
        counts = v['counts']
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
    precision, recall, f1 = get_stats(tp,fp,fn)
    #print(f"tp: {tp}")
    #print(f"fp: {fp}")
    #print(f"fn: {fn}")
    #print(f"precision: {precision}")
    #print(f"recall: {recall}")
    #print(f"F1: {f1}")
    return (precision, recall, f1), (tp,fp,fn)

@timeout(3, use_signals=False)
def get_query_results(prolog):
    results = next(iter(prolog.query("setof(X-Y, query(X,Y), Solutions)")))
    return results

if __name__ == '__main__':
    for preds in [1,2,4,8]:
        print(f"==== Scoring {preds}-Pred Background ====")
        prolog = Prolog()
        stack_limit_gb = 2
        stack_limit_bytes = stack_limit_gb * (1024**3)
        list(prolog.query(f"set_prolog_flag(stack_limit, {stack_limit_bytes})"))

        with open(f'../datasets/nonisomorphic_rules/prolog_graphs/bk_{preds}_random_preds.pl', 'r') as f:
            facts = f.readlines()
        for fact in facts:
            if fact.startswith('%'):
                continue
            else:
                prolog.assertz(fact.strip().strip('.'))

        with open(f"../results2/learned_rules_r{preds}.pl", "r") as f:
            rules = [l[:l.find('.')+1] for l in f.readlines() if l.strip() and not l.startswith('%')]

        rule_results = {}
        pbar = tqdm(rules)
        for rule in pbar:
            head, body = rule.split(" :- ")
            head = head.split("(")[0].strip() #)
            query = f"query(X,Y) :- " + body.strip()[:-1]
            prolog.assertz(query)
            gt_facts = set([tuple(r.values()) for r in prolog.query(f"{head}(X,Y)") if r])
            try:
                #results = next(iter(prolog.query("setof(X-Y, query(X,Y), Solutions)")))
                results = get_query_results(prolog)
                query_preds = set([tuple(s.strip('-()').split(', ')) for s in results['Solutions']])
                prolog.retract('query(X,Y) :- _')
                #query_preds = set([tuple(r.values()) for r in prolog.query("setof(X-Y, query(X,Y), Solutions)") if r])
            except Exception as e:
                #rule_results[head] = {'f1':0,'precision':0,'recall':0,'counts':(0,0,0)}
                query_preds = set()
                prolog.retract('query(X,Y) :- _')
            counts = score_sets(query_preds, gt_facts)
            precision, recall, f1 = get_stats(*counts)
            if head in rule_results:
                prev_f1 = rule_results[head]['f1']
                if f1>prev_f1:
                    rule_results[head] = {'f1':f1,'precision':precision,'recall':recall,'counts':counts}
            else:
                rule_results[head] = {'f1':f1,'precision':precision,'recall':recall,'counts':counts}
            #print(f1)
            stats,_ = print_scores(rule_results)
            pbar.set_description(f"(micro) F1: {round(stats[2],3)}, Precision: {round(stats[0],3)}, Recall: {round(stats[1],3)}")

        f1s = []
        ps = []
        rs = []
        for k,v in rule_results.items():
            counts = v['counts']
            p,r,f1 = get_stats(*counts)
            f1s.append(f1)
            ps.append(p)
            rs.append(r)
        print(f"Macro avg F1: {round(sum(f1s)/len(f1s),5)}")
        print(f"Macro avg Precision: {round(sum(ps)/len(ps),5)}")
        print(f"Macro avg Recall: {round(sum(rs)/len(rs),5)}")

        with open(f"scored_rules_ncrl_r{preds}.pkl", "wb") as f:
            pickle.dump(rule_results, f)
