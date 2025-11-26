python main.py --train --data kinship --model kinship --max_path_len 3 --anchor 10000

python main.py --test --get_rule --data kinship --model kinship --learned_path_len 2 --topk 500 --output_file ours_kinship
python main.py --test --get_rule --data kinship --model kinship --learned_path_len 3 --topk 500 --output_file ours_kinship

mkdir ours_kinship
mv ./ours_kinship_500* ./ours_kinship/

python kg_completion.py --data kinship --rule ours_kinship --top 8 --batch_size 1
