run script:
```
mv ../out/eventExtract/corpus_pred.jsonl ../data/graph/
python build_graph.py
coref_dir=../data/eventCoref
cp $coref_dir/event_corpus.json $coref_dir/events.json
cp $coref_dir/pred_corpus.csv $coref_dir/pred.csv
rm ../out/eventCoref/cached_dev_*

python ../eventCoreference/run_extraction.py \
--data_dir \
../data/eventCoref \
--output_dir \
../out/eventCoref \
--do_eval \
--fp16

cp ../out/eventCoref/pred_results.csv ../data/graph/corpus_coref_pred.csv
python build_graph.py
```

