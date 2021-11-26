import os
import sys
import json

gen_dir='../data/metaqa/'
sys.argv = [
        "",
        gen_dir,
    ]

from subgraph import main as subgraph_main

for prefix in ['1hop','2hop','3hop']:
    sys.argv = [
        "",
        gen_dir,
        f"{prefix}_"
    ]
    subgraph_main()
# os.system(f"cp {gen_dir}/test_ppr_all.jsonl {gen_dir}/test.jsonl")

# from answerGeneration.run_node_classification import main as gen_main
#
# sys.argv = [
#                "",
#                "--data_dir",
#                gen_dir,
#                "--output_dir",
#                gen_model_out_dir,
#                "--do_eval",
#                "--max_node_size",
#                "200",
#                "--block_ngram",
#                "5",
#                "--pred_dir",
#                gen_dir,
#                "--per_gpu_eval_batch_size",
#                "1",
#                 "--no_cuda",
#                 "--overwrite_cache"
#            ] # + common_args
# gen_main()

