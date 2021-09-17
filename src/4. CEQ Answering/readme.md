run script:
```
data_dir=../data/answerGeneration
cp ../data/graph/digraph.json $data_dir/
cp ../out/eventExtract/train_qa_pred.jsonl $data_dir/train_pred.jsonl
cp ../out/eventExtract/dev_qa_pred.jsonl $data_dir/dev_pred.jsonl
cp ../out/eventExtract/test_qa_pred.jsonl $data_dir/test_pred.jsonl
python event_link.py
coref_dir=../data/eventCoref
cp $coref_dir/event_link.json $coref_dir/events.json
cp $coref_dir/pred_link.csv $coref_dir/pred.csv
rm ../out/eventCoref/cached_dev_*

python ../eventCoreference/run_classification.py \
--data_dir \
../data/eventCoref \
--output_dir \
../out/eventCoref \
--do_eval \
--fp16

cp ../out/eventCoref/pred_results.csv $data_dir/link_pred.csv
python event_link.py
python subgraph.py
cp dev-test_ppr_all_new.jsonl dev.jsonl


train

-------test--------
python ../answerGeneration/run_node_classification.py \
--data_dir
../data/answerGeneration
--output_dir
../out/answerGeneration-d-b16
--max_node_size
200
--fp16
--block_ngram
5
--do_eval
```