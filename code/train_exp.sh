CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_exp.py configs/exp

