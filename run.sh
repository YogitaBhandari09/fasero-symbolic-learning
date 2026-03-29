python src/data/dataset_generator.py --num-samples 5000 --series-order 5 --seed 42
python src/data/preprocess.py --max-len 30
python src/training/train_lstm.py
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
