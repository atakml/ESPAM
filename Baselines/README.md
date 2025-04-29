# Installing Packages
For the required packages, please find the `requirements.txt`. We recommand the use of conda virtual environment to install the package. 
  

# Environment Variables and Preparation
The only environment variable to be set is `PYTHONPATH`
```
export PYTHONPATH=$PYTHONPATH:{xx}/GNN-explain/codebase/:{xx}/GNN-explain/codebase/ExplanationEvaluation:{xx}/GraphXAI/graphxai:{xx}/GraphXAI/:{xx}/GStarX
```
Where `{xx}` is the absolute path of the current directory.

Before executing the experiments it is neccessary to unzip the following zip file in its directory:
```
{xx}/GNN-explain/codebase/ExplanationEvaluation/datasets/pkls/big_files.zip
```

# Evaluation Instruction
To evaluate the explanations, for the all the explainers (including ESPAM) except `GStarX` and `SubGraphX`please refer to the notebook `explanations_evaluation.py`. For the two others please follow the below instruction (We have parallelized the task for these two explainers):
```
1. In the bs.py, select the dataset name at the line 10: 
	Dataset names: aids, ba2motifs, bbbp
2. Select the desired explainer at line 56:
	2.1 Explainer names: "subgraph" for SubGraphX and "gstar" for gstarx 
	2.2 Please note that for each of the above configuration the following paths must be created first:
	{xx}/{dataset name}/{explainer}/
	{xx}/{dataset name}/{explainer}index
3. Run python bs.py
4. ** To make sure that all the instances are explained, please repeat 3 several time.
5. In the merger.py please select the dataset name and explainer the same as those in 1.
6. In the file bseval.py select the same name of the dataset at line 160 the same as 1.
7. Select the dataset name once again at line 163
8. Select the explainer name the same as 2 at line 196.
9. Run python bseval.py
10. Result are in the following path:
	{dataset_name}_res_{exp_name}_dict.pkl 
```
