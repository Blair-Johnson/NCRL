python main.py --train --data umls --model umls --max_path_len 3 --anchor 10000

python main.py --test --get_rule --data umls --model umls --learned_path_len 2 --topk 500 --output_file ours_umls
python main.py --test --get_rule --data umls --model umls --learned_path_len 3 --topk 500 --output_file ours_umls

mkdir ours_umls
mv ./ours_umls_500* ./ours_umls/

python kg_completion.py --data umls --rule ours_umls --top 8 --batch_size 1
