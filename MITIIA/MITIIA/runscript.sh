source /opt/miniconda3/etc/profile.d/conda.sh
conda activate conda_singularity
export PYTHONPATH="${PYTHONPATH}:."
python GNN_Transformer/scripts/main.py --config GNN_Transformer/configs/config_template.yml
