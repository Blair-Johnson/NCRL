import re
from typing import List, Dict, Sequence, Optional
import torch

def _prolog_atom(name: str) -> str:
    """
    Return a valid Prolog atom for a relation name.
    If name contains disallowed characters, quote it.
    """
    if re.match(r"^[a-z][a-zA-Z0-9_]*$", name):
        return name
    return f"'{name}'"

def _rel_call(rel: str, a: str, b: str) -> str:
    """
    Build a Prolog predicate call for rel(a,b), handling inv_ by swapping args.
    inv_pred(a,b) => pred(b,a)
    """
    if rel.startswith("inv_"):
        base = rel[4:]
        return f"{_prolog_atom(base)}({b}, {a})"
    return f"{_prolog_atom(rel)}({a}, {b})"

def _body_calls(body_rels: List[str]) -> List[str]:
    """
    Build a chain of calls r1(X,Z1), r2(Z1,Z2), ..., rn(Zn-1,Y)
    with inv_ handling built into each call.
    """
    # Variables: X, Z1..Zk-1, Y
    vars_ = ["X"] + [f"Z{i}" for i in range(1, len(body_rels))] + ["Y"]
    calls = []
    for i, rel in enumerate(body_rels):
        calls.append(_rel_call(rel, vars_[i], vars_[i+1]))
    return calls

def _head_call(head_rel: str) -> str:
    """
    head(X,Y) with proper handling for inv_head.
    """
    return _rel_call(head_rel, "X", "Y")

def write_prolog_rules(
    *,
    head_rdict,
    candidate_rule: Dict[int, List[str]],
    rule_conf: Dict[int, torch.Tensor],
    allowed_head_idx: Sequence[int],
    topk: int,
    out_path: str,
    include_conf_as_comment: bool = True,
) -> None:
    """
    Write mined rules as Prolog to out_path.

    - head_rdict: dataset.get_head_relation_dict()
    - candidate_rule: {rule_len: ["r1|r2|...", ...]}
    - rule_conf: {rule_len: tensor [num_bodies, num_heads]}
    - allowed_head_idx: list of head indices to write
    - topk: top-k bodies per head
    - out_path: destination .pl file
    """
    lines = []
    lines.append("% Auto-generated Prolog rules")
    lines.append("% Variables: X and Y are the head arguments; Z1..Zk are intermediates.")
    lines.append("")

    # For reproducibility, write heads in a stable order
    allowed_head_idx = list(sorted(set(allowed_head_idx)))

    for rule_len, conf_mat in rule_conf.items():
        lines.append(f"% ---- Rule length: {rule_len} ----")
        # conf_mat: [num_bodies, num_heads]
        sorted_val, sorted_idx = torch.sort(conf_mat, dim=0, descending=True)
        num_bodies, _ = sorted_val.shape

        bodies_all = candidate_rule[rule_len]  # list of "r1|r2|..."
        for r in allowed_head_idx:
            head_rel = head_rdict.idx2rel[r]
            if head_rel == "None":
                continue
            head_str = _head_call(head_rel)

            # Emit top-k for this head
            k = min(topk, num_bodies)
            for j in range(k):
                conf = sorted_val[j, r].item()
                body_str = bodies_all[sorted_idx[j, r].item()]
                body_rels = [b for b in body_str.split("|") if b]  # relations
                if not body_rels:
                    # No-op body; skip
                    continue

                calls = _body_calls(body_rels)
                rule_line = f"{head_str} :- {', '.join(calls)}."
                if include_conf_as_comment:
                    rule_line += f"  % conf={conf:.6f}, len={rule_len}, head={head_rel}"
                lines.append(rule_line)
            lines.append("")  # blank line between heads

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
