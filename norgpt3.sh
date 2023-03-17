#!/bin/bash
# This example will start serving the NorGPT-3B model.

#--nproc_per_node: #GPUS_PER_NODE
DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=/data/all/models/nor_gpt3/latest_checkpointed_iteration.txt
VOCAB_FILE=/data/all/models/nor_gpt3/vocab.json
MERGE_FILE=/data/all/models/nor_gpt3/merges.txt
CUDA_HOME='/usr/local/cuda-11.4'
pip install flask-restful

#--temperature: Sampling temperature.
#--top_p: Top p sampling.
#--out-seq-length: Size of the output generated text.
python -m torch.distributed.run $DISTRIBUTED_ARGS ./tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 2688  \
       --load ${CHECKPOINT}  \
       --num-attention-heads 32  \
       --max-position-embeddings 2048  \
       --tokenizer-type GPT2BPETokenizer  \
       --fp16  \
       --micro-batch-size 4  \
       --seq-length 2048  \
#       --out-seq-length 1024  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top_p 0.9  \
       --seed 42
