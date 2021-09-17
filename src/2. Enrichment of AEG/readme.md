>export CUDA_VISIBLE_DEVICES=0

train example: (train.csv and pred.csv in data_dir)
```
python classification.py
--data_dir
../data/eventCoref
--output_dir
../out/eventCoref
--do_train
--evaluate_during_training
--fp16
--overwrite_output_dir
```

predict example:
```
python classification.py
--data_dir
../data/eventCoref
--output_dir
../out/eventCoref
--do_eval
--fp16
```

run script:
```
data_dir=../data/eventCoref

cp $data_dir/event_labeled.json $data_dir/events.json

python sample.py

cp $data_dir/dev.csv $data_dir/pred.csv

python classification.py \
--data_dir \
../data/eventCoref \
--output_dir \
../out/eventCoref \
--do_train \
--evaluate_during_training \
--fp16 \
--overwrite_output_dir
```