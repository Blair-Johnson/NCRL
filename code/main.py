import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample 
import random
from torch.nn.utils import clip_grad_norm_
import time
import pickle
import argparse
import numpy as np

from data import *
from utils import *
from model import *
from prolog_writer import write_prolog_rules
from torch.cuda.amp import autocast, GradScaler

import debugpy
# debugpy.listen(5695)
print("Waiting for debugger")
# debugpy.wait_for_client()
print("Attached! :)")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_msg(str(device))

rule_conf = {}
candidate_rule = {}

def parse_name_list(arg):
    if not arg:
        return None
    if arg.startswith('@'):
        with open(arg[1:], 'r') as f:
            items = [line.strip() for line in f if line.strip()]
    else:
        items = [x.strip() for x in arg.split(',') if x.strip()]
    return set(items)

def sample_training_data_old(max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict,
                         target_heads=None, body_rels=None, body_rdf_for_paths=None):
    print("Sampling training data...")
    anchors_rdf = []
    per_anchor_num = anchor_num//((head_rdict.__len__() -1) //2)
    fact_dict = construct_fact_dict(fact_rdf)

    for head in head_rdict.rel2idx:
        if head == "None" or "inv_" in head:
            continue
        if target_heads and head not in target_heads:
            continue
        sampled_rdf = sample_anchor_rdf(fact_dict.get(head, []), num=per_anchor_num)
        anchors_rdf.extend(sampled_rdf)

    print("Total_anchor_num", len(anchors_rdf))
    len2train_rule_idx, sample_number = {}, 0

    # Use only background predicates to construct paths
    graph_for_paths = body_rdf_for_paths if body_rdf_for_paths is not None else fact_rdf
    for anchor_rdf in anchors_rdf:
        rule_seq, record = construct_rule_seq(graph_for_paths, anchor_rdf, entity2desced, max_path_len, PRINT=False)
        sample_number += len(record)
        if rule_seq:
            for rule in rule_seq:
                # Optionally filter out any rule whose body uses predicates outside body_rels
                if body_rels:
                    body_str = rule.split('-')[0]
                    body_ok = all((r in body_rels) for r in body_str.split('|'))
                    if not body_ok:
                        continue
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                body_len = len(idx) - 2
                len2train_rule_idx.setdefault(body_len, []).append(idx)

    print("# train_rule examples:", sample_number)
    print("Fact set number:{} Sample number:{}".format(len(fact_rdf), sample_number))
    for rule_len in len2train_rule_idx:
        print("sampled examples for rule of length {}: {}".format(rule_len, len(len2train_rule_idx[rule_len])))
    return len2train_rule_idx

def sample_training_data(max_path_len, anchor_num, head_triplets_by_head, entity2desced_body, entity2desced_head, head_rdict):
    """
    head_triplets_by_head: dict[head_rel] -> list of (h, head_rel, t) from train set (positives)
    entity2desced_body: traversal graph over background (body) relations
    entity2desced_head: direct-link graph over target head relations (for connected())
    """
    print("Sampling training data...")

    target_heads = [h for h in head_triplets_by_head.keys() if h != "None" and "inv_" not in h]
    n_target_heads = max(1, len(target_heads))
    per_anchor_num = anchor_num // n_target_heads

    print(f"Number of target head relations:{n_target_heads}")
    print(f"Number of per_anchor_num: {per_anchor_num}")

    anchors_rdf = []
    for head in target_heads:
        pool = head_triplets_by_head.get(head, [])
        chosen = sample_anchor_rdf(pool, num=per_anchor_num) if pool else []
        anchors_rdf.extend(chosen)

    print("Total_anchor_num", len(anchors_rdf))

    len2train_rule_idx = {}
    sample_number = 0

    debug_dump = 3

    for anchor_rdf in anchors_rdf:
        rule_seq, record = construct_rule_seq(
            anchor_rdf=anchor_rdf,
            entity2desced_body=entity2desced_body,
            entity2desced_head=entity2desced_head,
            max_path_len=max_path_len,
            PRINT=False
        )

        # Count actual mined rule strings
        sample_number += len(rule_seq)

        if debug_dump > 0:
            ah, ar, at = parse_rdf(anchor_rdf)
            print(f"[anchor] {ah} --{ar}--> {at}  mined_rules={len(rule_seq)}")
            for r in rule_seq[:5]:
                print(f"    rule: {r}")
            debug_dump -= 1

        for rule in rule_seq:
            idx = torch.LongTensor(rule2idx(rule, head_rdict))
            body_len = len(idx) - 2
            len2train_rule_idx.setdefault(body_len, []).append(idx)

    print(f"# train_rule examples: {sample_number}")
    rule_len_range = list(len2train_rule_idx.keys())
    print("Fact set number is not used here (anchors from train positives).")
    for rule_len in rule_len_range:
        print(f"sampled examples for rule of length {rule_len}: {len(len2train_rule_idx[rule_len])}")
    return len2train_rule_idx

        
def train(args, dataset):
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()

    # Parse user-provided subsets
    target_heads = parse_name_list(getattr(args, "target_heads", "")) or set()
    body_rels = parse_name_list(getattr(args, "body_rels", "")) or set()

    # Build head graph (direct links) from TRAIN positives for target heads
    train_rdf = dataset.train_rdf
    head_rdf_for_heads = [rdf for rdf in train_rdf if (not target_heads or parse_rdf(rdf)[1] in target_heads)]
    # Dict for anchor sampling
    head_triplets_by_head = construct_fact_dict(head_rdf_for_heads)
    entity2desced_head = construct_descendant(head_rdf_for_heads)

    # Build body traversal graph from background relations
    # If your facts.txt already only includes background relations, using dataset.fact_rdf is fine.
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    if body_rels:
        body_rdf = [rdf for rdf in all_rdf if parse_rdf(rdf)[1] in body_rels]
    else:
        body_rdf = dataset.fact_rdf  # typical background
    entity2desced_body = construct_descendant(body_rdf)

    # Sanity checks
    if not head_rdf_for_heads:
        print("WARNING: No head triples found in train set for the provided target_heads. No anchors will be sampled.")
    if not entity2desced_body:
        print("WARNING: Body traversal graph is empty. No paths can be mined. Check --body_rels and facts.")

    # Sample training data and mine rules
    max_path_len = args.max_path_len
    anchor_num = args.anchor
    len2train_rule_idx = sample_training_data(
        max_path_len=max_path_len,
        anchor_num=anchor_num,
        head_triplets_by_head=head_triplets_by_head,
        entity2desced_body=entity2desced_body,
        entity2desced_head=entity2desced_head,
        head_rdict=head_rdict
    )

    print_msg("  Start training  ")
    batch_size = 5000
    emb_size = 1024
    n_epoch = 1500
    lr = 0.000025

    model = Encoder(rdict.__len__(), emb_size, device)
    if torch.cuda.is_available():
        model = model.cuda()

    loss_func_head = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    amp_enabled = (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled and args.mixed_precision)
    model.train()
    train_acc = {}

    # Iterate only over lengths we actually have
    body_len_range = sorted(len2train_rule_idx.keys())
    if not body_len_range:
        print("No training examples were generated. Try increasing --anchor, relaxing --body_rels, or verifying inv_ edges.")
        return

    print("body_len_range", body_len_range)

    for rule_len in body_len_range:
        rule_ = len2train_rule_idx.get(rule_len, [])
        if not rule_:
            print(f"Skipping rule length {rule_len}: 0 training examples")
            continue

        print(f"\nrule length:{rule_len}")
        train_acc[rule_len] = []

        for epoch in range(n_epoch):
            model.zero_grad()
            sample_rule_ = sample(rule_, batch_size) if len(rule_) > batch_size else rule_
            body_ = [r_[0:-2] for r_ in sample_rule_]
            head_ = [r_[-1] for r_ in sample_rule_]

            inputs_h = torch.stack(body_, 0).to(device)
            targets_h = torch.stack(head_, 0).to(device)

            with autocast(enabled=amp_enabled):
                pred_head, _entropy_loss = model(inputs_h)
                loss_head = loss_func_head(pred_head, targets_h.reshape(-1))
                entropy_loss = _entropy_loss.mean()
                loss = args.alpha * loss_head + (1 - args.alpha) * entropy_loss

            if epoch % (n_epoch // 10) == 0:
                print("### epoch:{}\tloss_head:{:.3}\tentropy_loss:{:.3}\tloss:{:.3}\t".format(epoch, loss_head, entropy_loss, loss))

            train_acc[rule_len].append(((pred_head.argmax(dim=1) == targets_h.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy())

            clip_grad_norm_(model.parameters(), 0.5)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

    print("Saving model...")
    with open('../results/{}'.format(args.model), 'wb') as g:
        pickle.dump(model, g)

def enumerate_body(relation_num, rdict, body_len):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(range(relation_num), repeat=body_len))
    # transfer index to relation name
    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body

def enumerate_body_subset(allowed_idx, rdict, body_len):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(allowed_idx, repeat=body_len))
    idx2rel = rdict.idx2rel
    all_body = [[idx2rel[x] for x in b_idx_] for b_idx_ in all_body_idx]
    return all_body_idx, all_body

def test(args, dataset):
    head_rdict = dataset.get_head_relation_dict()

    # Parse user subsets
    allowed_heads = parse_name_list(getattr(args, "target_heads", "")) or None
    allowed_body = parse_name_list(getattr(args, "body_rels", "")) or None

    # Build list of allowed head indices (exclude None and inv_)
    if allowed_heads:
        allowed_head_idx = [i for i, r in head_rdict.idx2rel.items()
                            if r in allowed_heads]
    else:
        allowed_head_idx = [i for i, r in head_rdict.idx2rel.items()
                            if r != "None" and not r.startswith("inv_")]

    if len(allowed_head_idx) == 0:
        print("WARNING: No allowed heads resolved from --target_heads; rule extraction will be empty.")
    else:
        print("Allowed head relations for rule extraction:")
        for i in allowed_head_idx:
            print("  -", head_rdict.idx2rel[i])

    # Build allowed body relation indices
    if allowed_body:
        allowed_body_idx = [head_rdict.rel2idx[r] for r in allowed_body if r in head_rdict.rel2idx]
        print("Allowed body relations:")
        for r in allowed_body:
            print("  -", r, "(known)" if r in head_rdict.rel2idx else "(UNKNOWN NAME)")
        if not allowed_body_idx:
            print("WARNING: No allowed body relations mapped to indices; falling back to all relations.")
    else:
        allowed_body_idx = None

    with open('../results/{}'.format(args.model), 'rb') as g:
        if torch.cuda.is_available():
            model = pickle.load(g)
            model.to(device)
        else:
            model = torch.load(g, map_location='cpu')

    print_msg("  Start Eval  ")
    model.eval()

    amp_enabled = (device.type == 'cuda') and args.mixed_precision

    # Enumerate bodies
    r_num = head_rdict.__len__()-1
    batch_size = 1000
    rule_len = args.learned_path_len
    print("\nrule length:{}".format(rule_len))

    probs = []
    if allowed_body_idx:
        _, body = enumerate_body_subset(allowed_body_idx, head_rdict, body_len=rule_len)
    else:
        _, body = enumerate_body(r_num, head_rdict, body_len=rule_len)

    body_list = ["|".join(b) for b in body]
    candidate_rule[rule_len] = body_list
    n_epoches = math.ceil(float(len(body_list))/ batch_size)

    # Precompute head mask vector to suppress non-allowed heads
    if allowed_head_idx:
        head_mask = torch.full((head_rdict.__len__(),), -1e9, device=device)
        for idx in allowed_head_idx:
            head_mask[idx] = 0.0
    else:
        head_mask = None

    for epoches in range(n_epoches):
        if epoches == n_epoches-1:
            bodies = body_list[epoches*batch_size:]
        else:
            bodies = body_list[epoches*batch_size: (epoches+1)*batch_size]
            
        body_idx = body2idx(bodies, head_rdict)
        inputs = torch.LongTensor(np.array(body_idx)).to(device) if torch.cuda.is_available() else torch.LongTensor(np.array(body_idx))
        print("## body {}".format((epoches+1)* batch_size))
            
        with torch.no_grad():
            with autocast(enabled=amp_enabled):
                pred_head, _entropy_loss = model(inputs)  # [batch_size, 2*n_rel+1]
            # Mask out non-allowed heads before softmax so probabilities are among allowed only
            if head_mask is not None and pred_head.shape[1] == head_mask.shape[0]:
                pred_head = pred_head + head_mask.to(pred_head.dtype)  # add -1e9 to disallowed columns
            prob_ = torch.softmax(pred_head, dim=-1)
            probs.append(prob_.detach().cpu())
      
    rule_conf[rule_len] = torch.cat(probs,dim=0)
    print ("rule_conf",rule_conf[rule_len].shape)

if __name__ == '__main__':
    msg = "First Order Logic Rule Mining"
    print_msg(msg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="increase output verbosity")
    parser.add_argument("--test", action="store_true", help="increase output verbosity")
    parser.add_argument("--get_rule", action="store_true", help="increase output verbosity")
    parser.add_argument("--data", default="family", help="increase output verbosity")
    parser.add_argument("--topk", type=int, default=200, help="increase output verbosity")
    parser.add_argument("--anchor", type=int, default=10000, help="increase output verbosity")
    parser.add_argument("--gpu", type=int, default=1, help="increase output verbosity")
    parser.add_argument("--output_file", default="ours_family", help="increase output verbosity")
    parser.add_argument("--model", default="family", help="increase output verbosity")
    parser.add_argument("--max_path_len", type=int, default=4, help="increase output verbosity")
    parser.add_argument("--learned_path_len", type=int, default=4, help="increase output verbosity")
    parser.add_argument("--sparsity", type=float, default=1, help="increase output verbosity")
    parser.add_argument("--alpha", type=float, default=1, help="increase output verbosity")
    parser.add_argument("--target_heads", default="", help="Comma-separated list or @file of head predicates to learn")
    parser.add_argument("--body_rels", default="", help="Comma-separated list or @file of predicates allowed in rule bodies")
    parser.add_argument("--prolog_out", default="", help="If set, also write rules to a Prolog file at this path (e.g., ./rules.pl)")
    parser.add_argument("--mixed_precision", action="store_true", help="enable mixed precision training")

    args = parser.parse_args()
    assert args.train or args.test

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # DataSet
    data_path = '../datasets/{}/'.format(args.data)
    dataset = Dataset(data_root=data_path, sparsity=args.sparsity, inv=True)
    print("Dataset:{}".format(data_path)) 

    target_heads = parse_name_list(args.target_heads)
    body_rels = parse_name_list(args.body_rels)
    
    # Saved Model
    model_path = '../results/{}'.format(args.model)
    print("Model:{}".format(model_path))

    if args.train:
        print_msg("Train!")
        train(args, dataset)

    if args.test:
        print_msg("Test!")
        test(args, dataset)
            
        if args.get_rule:
            print_msg("Generate Rule!")
            
            head_rdict = dataset.get_head_relation_dict()

            # Reuse the same allowed head filter here for writing
            allowed_heads = parse_name_list(getattr(args, "target_heads", "")) or None
            if allowed_heads:
                allowed_head_idx = [i for i, r in head_rdict.idx2rel.items() if r in allowed_heads]
            else:
                allowed_head_idx = [i for i, r in head_rdict.idx2rel.items() if r != "None" and not r.startswith("inv_")]
            if not allowed_head_idx:
                print("WARNING: no heads to write; check --target_heads")
            
            for rule_len in rule_conf:
                rule_path = "./{}_{}_{}.txt".format(args.output_file, args.topk, rule_len)
                print("\nrule length:{}".format(rule_len))
                sorted_val, sorted_idx = torch.sort(rule_conf[rule_len], 0, descending=True)

                n_rules, n_heads = sorted_val.shape
                print(f"Writing heads: {[head_rdict.idx2rel[i] for i in allowed_head_idx]}")

                with open(rule_path, 'w') as g:
                    for r in allowed_head_idx:
                        head = head_rdict.idx2rel[r]
                        idx = 0
                        # Take top-k bodies for this head
                        while idx<args.topk and idx<n_rules:
                            conf = sorted_val[idx, r].item()
                            body = candidate_rule[rule_len][sorted_idx[idx, r].item()]
                            msg = "{:.3f} ({:.3f})\t{} <-- ".format(conf, conf, head)
                            body = body.split('|')
                            msg += ", ".join(body)
                            g.write(msg + '\n')
                            idx+=1
            if args.prolog_out:
                print_msg(f"Writing Prolog rules to {args.prolog_out}")
                write_prolog_rules(
                    head_rdict=head_rdict,
                    candidate_rule=candidate_rule,
                    rule_conf=rule_conf,
                    allowed_head_idx=allowed_head_idx,
                    topk=args.topk,
                    out_path=args.prolog_out,
                )
