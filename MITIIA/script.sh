source /opt/miniconda3/etc/profile.d/conda.sh
cd /mnt/MITIIA
conda activate conda_singularity
pip install -e .
export PYTHONPATH=":/mnt/MITIIA/GNN_Transformer/optimal_transport_for_gnn/src/ :/mnt/MITIIA/GNN_Transformer/optimal_transport_for_gnn/src/fgw_ot :/mnt/MITIIA/GNN_Transformer/optimal_transport_for_gnn/src/"
