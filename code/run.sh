BACKGROUND_PREDS[1]="di_edge_0,inv_di_edge_0"
BACKGROUND_PREDS[2]="di_edge_0,inv_di_edge_0,di_edge_1,inv_di_edge_1"
BACKGROUND_PREDS[4]="di_edge_0,inv_di_edge_0,di_edge_1,inv_di_edge_1,di_edge_2,inv_di_edge_2,di_edge_3,inv_di_edge_3"
BACKGROUND_PREDS[8]="di_edge_0,inv_di_edge_0,di_edge_1,inv_di_edge_1,di_edge_2,inv_di_edge_2,di_edge_3,inv_di_edge_3,di_edge_4,inv_di_edge_4,di_edge_5,inv_di_edge_5,di_edge_6,inv_di_edge_6,di_edge_7,inv_di_edge_7"

ANCHOR[1]="64"
ANCHOR[2]="128"
ANCHOR[4]="256"
ANCHOR[8]="512"

for NPREDS in 1 2 4 8
do
    python main.py --train --data "nonisomorphic_rules/tsv/r${NPREDS}" --max_path_len 5 --model "r${NPREDS}_model" --gpu 0 --get_rule --topk 128 --target_heads "@../datasets/nonisomorphic_rules/tsv/r${NPREDS}/label_relations.txt" --body_rels "${BACKGROUND_PREDS[${NPREDS}]}" --output_file "r${NPREDS}_ckpt" --test --prolog_out "../results2/learned_rules_r${NPREDS}.pl" --mixed_precision --per_head_loop --anchor "${ANCHOR[${NPREDS}]}"
done
