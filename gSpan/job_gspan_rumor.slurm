#!/bin/bash

#SBATCH -J get_embeddings          ## job name  任务名称，建议英文
#SBATCH -p compute         ## partion name 队列名称
#SBATCH -N 1               ## number of hosts 节点个数
#SBATCH -n 32               ## number  of cores  cpu核数  建议GPU:CPU配比 1:6
#SBATCH --gres=gpu:v100:0  ## number of GPUS  每个节点GPU卡数


module load conda3
source activate pyg37
free -h
which python
cd /share/home/ypw_test/gSpan/
echo "python -m gspan_mining -s 10 -l 3 -u 3 -p True -w True -n 1000 ./graphdata/graph_gspan_rumor.data"
python -m gspan_mining -s 10 -l 3 -u 3 -p True -w True -n 1000 ./graphdata/graph_gspan_rumor.data
conda deactivate

