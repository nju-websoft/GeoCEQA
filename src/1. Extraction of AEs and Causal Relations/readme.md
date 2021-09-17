24000MB memory

train example:
```
python run_extraction.py
--data_di
../data/eventExtract
--output_dir
../out/eventExtract
--do_train
--evaluate_during_training
--fp16
--overwrite_output_dir
```

predict example:
```
python run_extraction.py
--data_dir
../data/eventExtract
--output_dir
../out/eventExtract
--do_predict
--predict_file
../data/eventExtract/corpus.jsonl
--fp16
```

pipeline run script:
```
python run_extraction.py \
--data_dir \
../data/eventExtract \
--output_dir \
../out/eventExtract \
--do_train \
--evaluate_during_training \
--fp16 \
--overwrite_output_dir

python run_extraction.py \
--data_dir \
../data/eventExtract \
--output_dir \
../out/eventExtract \
--do_predict \
--predict_file \
../data/eventExtract/corpus.jsonl \
--fp16

python run_extraction.py \
--data_dir \
../data/eventExtract \
--output_dir \
../out/eventExtract \
--do_predict \
--predict_file \
../data/eventExtract/train_qa.jsonl \
--fp16

python run_extraction.py \
--data_dir \
../data/eventExtract \
--output_dir \
../out/eventExtract \
--do_predict \
--predict_file \
../data/eventExtract/dev_qa.jsonl \
--fp16

python run_extraction.py \
--data_dir \
../data/eventExtract \
--output_dir \
../out/eventExtract \
--do_predict \
--predict_file \
../data/eventExtract/test_qa.jsonl \
--fp16
```
