## Requirements

### Environment:

- Python 3.6
- ubuntu 18.04
- TiTan
- Cuda 11.1

### Packages

- [pytorch 1.2.0](https://pytorch.org/get-started/previous-versions/)
- [numpy 1.17.4](https://numpy.org/install/)
- [tensorboardX 1.9](https://github.com/lanpa/tensorboardX)
- [transformers 2.2.2](https://huggingface.co/transformers/installation.html)
- [fp16](https://github.com/NVIDIA/apex)
- [sklearn 0.23.1](https://scikit-learn.org/stable/install.html)
- [rouge 0.3.1](https://pypi.org/project/rouge/)
- [nltk 3.4.5](https://pypi.org/project/nltk/)

## Quickstart
unzip AEG.zip,dev.zip,test.zip,train.zip in Data/Corpus and AEG

### Extraction of AEs and Causal Relations

#### Training

```bash
python run_extraction.py --data_dir "../../Data/Training-Validation-Test splits/Extraction of AEs and Causal Relations" --output_dir "../../out/Extraction of AEs and Causal Relations" --do_train --evaluate_during_training --overwrite_output_dir --fp16
```

#### Predict

```bash
python run_extraction.py --data_dir "../../Data/Training-Validation-Test splits/Extraction of AEs and Causal Relations" --output_dir "../../out/Extraction of AEs and Causal Relations" --do_predict --predict_file "../../Data/Corpus and AEG/Corpus.jsonl" --fp16
```

output file is "Corpus_pred.jsonl" in output_dir

### Enrichment of AEG

#### Training

```bash
python run_classification.py --data_dir "../../Data/Training-Validation-Test splits/Enrichment of AEG" --output_dir "../../out/Enrichment of AEG" --do_train --evaluate_during_training --fp16 --overwrite_output_dir
```

### Construction of AEG

#### step 1:  generate candidate AE pairs

```bash
python build_graph.py --input_path "../../out/Extraction of AEs and Causal Relations/" --output_path "../../out/graph/"
```

#### step 2: predict relations

```bash
python run_classification.py --data_dir "../../out/graph/" --output_dir "../../out/Enrichment of AEG" --do_eval --fp16
```

#### step 3: construct AEG

```bash
cp "../../out/Extraction of AEs and Causal Relations/Corpus_pred.jsonl"  "../../out/Enrichment of AEG/Corpus_pred.jsonl"

python build_graph.py --input_path "../../out/Enrichment of AEG/" --output_path "../../out/graph/"
```

AEG.json will be generated in output_path

### CEQ Answering

fasttext file could be download from [link](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz) and put in src/4. CEQ Answering/

#### step 1: extract AEs and Causal Relations for train dev test

```bash
python run_extraction.py --data_dir "../../Data/Training-Validation-Test splits/Extraction of AEs and Causal Relations" --output_dir "../../out/Extraction of AEs and Causal Relations" --do_predict --predict_file "../../Data/Training-Validation-Test splits/CEQ Answering/train.jsonl" --fp16

python run_extraction.py --data_dir "../../Data/Training-Validation-Test splits/Extraction of AEs and Causal Relations" --output_dir "../../out/Extraction of AEs and Causal Relations" --do_predict --predict_file "../../Data/Training-Validation-Test splits/CEQ Answering/dev.jsonl" --fp16

python run_extraction.py --data_dir "../../Data/Training-Validation-Test splits/Extraction of AEs and Causal Relations" --output_dir "../../out/Extraction of AEs and Causal Relations" --do_predict --predict_file "../../Data/Training-Validation-Test splits/CEQ Answering/test.jsonl" --fp16
```

#### step 2: link AEs

```bash
python event_link.py "../../out/graph/" "../../out/CEQ Answering/" "../../out/Extraction of AEs and Causal Relations/dev_pred.jsonl" "../../out/CEQ Answering/extractions" "dev"
python event_link.py "../../out/graph/" "../../out/CEQ Answering/" "../../out/Extraction of AEs and Causal Relations/train_pred.jsonl" "../../out/CEQ Answering/extractions" "train"
python event_link.py "../../out/graph/" "../../out/CEQ Answering/" "../../out/Extraction of AEs and Causal Relations/test_pred.jsonl" "../../out/CEQ Answering/extractions" "test"

python run_classification.py --data_dir "../../out/CEQ Answering/extractions_train/" --output_dir "../../out/Enrichment of AEG" --pred_dir "../../out/CEQ Answering/extractions_train/" --do_eval --fp16
python run_classification.py --data_dir "../../out/CEQ Answering/extractions_dev/" --output_dir "../../out/Enrichment of AEG" --pred_dir "../../out/CEQ Answering/extractions_dev/" --do_eval --fp16
python run_classification.py --data_dir "../../out/CEQ Answering/extractions_test/" --output_dir "../../out/Enrichment of AEG" --pred_dir "../../out/CEQ Answering/extractions_test/" --do_eval --fp16

python event_link.py "../../out/graph/" "../../out/CEQ Answering/" "../../out/Extraction of AEs and Causal Relations/test_pred.jsonl" "../../out/CEQ Answering/extractions" "test"
python event_link.py "../../out/graph/" "../../out/CEQ Answering/" "../../out/Extraction of AEs and Causal Relations/train_pred.jsonl" "../../out/CEQ Answering/extractions" "train"
python event_link.py "../../out/graph/" "../../out/CEQ Answering/" "../../out/Extraction of AEs and Causal Relations/dev_pred.jsonl" "../../out/CEQ Answering/extractions" "dev"
```

#### step 3: extract subgraph

```bash
python subgraph.py "../../out/graph/" "../../out/CEQ Answering/extractions_train/" "../../out/CEQ Answering/"
python subgraph.py "../../out/graph/" "../../out/CEQ Answering/extractions_dev/" "../../out/CEQ Answering/"
python subgraph.py "../../out/graph/" "../../out/CEQ Answering/extractions_test/" "../../out/CEQ Answering/"
```

#### step 4: training

AEG.json and train,dev,test datset after subgraph extraction should be put in data_dir

```bash
python run_node_classification.py --data_dir "../../Data/Corpus and AEG" --output_dir ../../out/answerGeneration-d-b16-n --max_node_size 200 --fp16 --do_train --evaluate_during_training --overwrite_output_dir 
```

#### step 5: predict

model after training could be download from [link](https://drive.google.com/file/d/1-j8FvkYAUkjQccmkHChdlBbU2mjQBenG/view?usp=sharing)

```bash
python run_node_classification.py --data_dir "../../Data/Corpus and AEG" --output_dir ../../out/answerGeneration-d-b16-n --max_node_size 200 --fp16 --block_ngram 5 --do_eval
```

