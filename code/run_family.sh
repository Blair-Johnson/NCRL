python main.py --train --data family --model family --max_path_len 3 --anchor 10000

python main.py --test --get_rule --data family --model family --learned_path_len 2 --topk 500 --output_file ours_family
python main.py --test --get_rule --data family --model family --learned_path_len 3 --topk 500 --output_file ours_family

mkdir ours_family
mv ./ours_family_500* ./ours_family/

python kg_completion.py --data family --rule ours_family --top 8 --batch_size 1
