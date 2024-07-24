# Advanced-NLP-Classification-Integrating-BERT
## Lessons

Learn how to combine machine learning with software engineering to design, develop, deploy and iterate on production-grade ML applications.
## Overview

In this repositry, we'll go from experimentation (design + development) to production (deployment + iteration). We'll do this iteratively by motivating the components that will enable us to build a *reliable* production system.


<br>

- **💡 First principles**: before we jump straight into the code, we develop a first principles understanding for every machine learning concept.
- **💻 Best practices**: implement software engineering best practices as we develop and deploy our machine learning models.
- **📈 Scale**: easily scale ML workloads (data, train, tune, serve) in Python without having to learn completely new languages.
- **⚙️ MLOps**: connect MLOps components (tracking, testing, serving, orchestration, etc.) as we build an end-to-end machine learning system.
- **🚀 Dev to Prod**: learn how to quickly and reliably go from development to production without any changes to our code or infra management.
- **🐙 CI/CD**: learn how to create mature CI/CD workflows to continuously train and deploy better models in a modular way that integrates with any stack.

## Audience

Machine learning is not a separate industry, instead, it's a powerful way of thinking about data that's not reserved for any one type of person.

- **👩‍💻 All developers**: whether software/infra engineer or data scientist, ML is increasingly becoming a key part of the products that you'll be developing.
- **👩‍🎓 College graduates**: learn the practical skills required for industry and bridge gap between the university curriculum and what industry expects.
- **👩‍💼 Product/Leadership**: who want to develop a technical foundation so that they can build amazing (and reliable) products powered by machine learning.

### Cluster

We'll start by setting up our cluster with the environment and compute configurations.

<details>
  <summary>Local</summary><br>
  Your personal laptop (single machine) will act as the cluster, where one CPU will be the head node and some of the remaining CPU will be the worker nodes. All of the code in this course will work in any personal laptop though it will be slower than executing the same workloads on a larger cluster.
</details>


### Git setup

Create a repository by following these instructions: [Create a new repository](https://github.com/new) → name it `Made-With-ML` → Toggle `Add a README file` (**very important** as this creates a `main` branch) → Click `Create repository` (scroll down)

Now we're ready to clone the repository that has all of our code:

```bash
git clone https://github.com/buntyk187/Advanced-NLP-Classification-Integrating-BERT.git .
```



### Credentials

```bash
touch .env
```
```bash
# Inside .env
GITHUB_USERNAME="CHANGE_THIS_TO_YOUR_USERNAME"  # ← CHANGE THIS
```
```bash
source .env
```

### Virtual environment

<details>
  <summary>Local</summary><br>

  ```bash
  export PYTHONPATH=$PYTHONPATH:$PWD
  python3 -m venv venv  # recommend using Python 3.10
  source venv/bin/activate  # on Windows: venv\Scripts\activate
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install -r requirements.txt
  pre-commit install
  pre-commit autoupdate
  ```

  > Highly recommend using Python `3.10` and using [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows).

</details>

## Notebook

Start by exploring the [jupyter notebook](notebooks/fine_tunning_NLP.ipynb) to interactively walkthrough the core machine learning workloads.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/systems-design/workloads.png">
</div>

<details>
  <summary>Local</summary><br>

  ```bash
  # Start notebook
  jupyter lab notebooks/fine_tunning_NLP.ipynb
```

</details>


## Scripts

Now we'll execute the same workloads using the clean Python scripts following software engineering best practices (testing, documentation, logging, serving, versioning, etc.) The code we've implemented in our notebook will be refactored into the following scripts:

```bash
madewithml
├── config.py
├── data.py
├── evaluate.py
├── models.py
├── predict.py
├── serve.py
├── train.py
├── tune.py
└── utils.py
```

**Note**: Change the `--num-workers`, `--cpu-per-worker`, and `--gpu-per-worker` input argument values below based on your system's resources. For example, if you're on a local laptop, a reasonable configuration would be `--num-workers 6 --cpu-per-worker 1 --gpu-per-worker 0`.

### Training
```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/bunty187/Advanced-NLP-Classification-Integrating-BERT-with-Ray-FastAPI-and-MLFLow/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python madewithml/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 1 \
    --cpu-per-worker 3 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/training_results.json
```

### Tuning
```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/bunty187/Advanced-NLP-Classification-Integrating-BERT-with-Ray-FastAPI-and-MLFLow/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"
python madewithml/tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 2 \
    --num-workers 1 \
    --cpu-per-worker 3 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/tuning_results.json
```

### Experiment tracking

We'll use [MLflow](https://mlflow.org/) to track our experiments and store our models and the [MLflow Tracking UI](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) to view our experiments. We have been saving our experiments to a local directory but note that in an actual production setting, we would have a central location to store all of our experiments. It's easy/inexpensive to spin up your own MLflow server for all of your team members to track their experiments on or use a managed solution like [Weights & Biases](https://wandb.ai/site), [Comet](https://www.comet.ml/), etc.

```bash
export MODEL_REGISTRY=$(python -c "from madewithml import config; print(config.MODEL_REGISTRY)")
mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri $MODEL_REGISTRY
```

<details>
  <summary>Local</summary><br>

  If you're running this notebook on your local laptop then head on over to <a href="http://localhost:8080/" target="_blank">http://localhost:8080/</a> to view your MLflow dashboard.

</details>


### Evaluation
```bash
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
export HOLDOUT_LOC="https://raw.githubusercontent.com/bunty187/Advanced-NLP-Classification-Integrating-BERT-with-Ray-FastAPI-and-MLFLow/main/datasets/holdout.csv"
python madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --results-fp results/evaluation_results.json
```
```json
{
  "timestamp": "July 3, 2024 09:26:18 AM",
  "run_id": "6149e3fec8d24f1492d4a4cabd5c06f6",
  "overall": {
    "precision": 0.9076136428670714,
    "recall": 0.9057591623036649,
    "f1": 0.9046792827719773,
    "num_samples": 191.0
  },
...
```

### Inference
```bash
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python scripts/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python madewithml/predict.py predict \
    --run-id $RUN_ID \
    --title "Transfer learning with transformers" \
    --description "Using transformers for transfer learning on text classification tasks."
```
```json
[{
  "prediction": [
    "natural-language-processing"
  ],
  "probabilities": {
    "computer-vision": 0.0009767753,
    "mlops": 0.0008223939,
    "natural-language-processing": 0.99762577,
    "other": 0.000575123
  }
}]
```
