# spi_pomdp
Safe Policy Improvement in Partially Observable Environments


After cloning this repository:

1. create a virtualenv and activate it
```bash
cd spi_pomdp/
python3 -m venv .venv
source .venv/bin/activate
```
2. install the dependencies
```bash
pip install -r requirements.txt
cd ../pogym/
pip install -e .
cd ../spi_pomdp/
```
3. run the unittests
```bash
python -m unittest discover
```
4. run all experiments
```bash
bash run.sh
```
5. generate the plots
```bash
cd plotting/
bash run.sh
```


The following sections give an overview of how the codebase can be used to run new experiments.

## Training a baseline policies:

```bash
$ python training_baseline_policy.py --env_id "Tiger-v0" --k 1 --training_episodes 5000 --decaying_rate_qlearning 0.002 --beta 0.05
```


```bash
$ python training_baseline_policy.py --help
usage: training_baseline_policy.py [-h] [--env_id {CheeseMaze-v0,Tiger-v0,Voicemail-v0}] [--training_episodes TRAINING_EPISODES] [--evaluation_episodes EVALUATION_EPISODES] [--decaying_rate_qlearning DECAYING_RATE_QLEARNING] [--beta BETA] [--k K]
                                   [--ma_size MA_SIZE] [--seed SEED] [--discount DISCOUNT]

optional arguments:
  -h, --help            show this help message and exit
  --env_id {CheeseMaze-v0,Tiger-v0,Voicemail-v0}
                        gym env id (default=Tiger-v0)
  --training_episodes TRAINING_EPISODES
                        number of training episodes (default=5000)
  --evaluation_episodes EVALUATION_EPISODES
                        number of evaluation episodes (default=10000)
  --decaying_rate_qlearning DECAYING_RATE_QLEARNING
                        q-learning exponential decay rate (default=0.002)
  --beta BETA           softmax temperature to generate the behavior policy (default=0.002)
  --k K                 size of history the FSC is tracking (default=1)
  --ma_size MA_SIZE     size of window for moving average -- percentage of the number of training episodes, (default=0.05)
  --seed SEED           seed (default=123)
  --discount DISCOUNT   discount factor -- gamma (default=0.95)
```

## Run SPI experiments


The following command runs the safe policy improvement experiments with seeds `1-16` using `8` using the behavior policy from the file `data/Tiger-v0_k-1.pkl`

```bash
$ python main.py -c 8 -s $(seq -s \  1 16) -p data/Tiger-v0_k-1.pkl
```


```bash
$ python main.py --help
usage: main.py [-h] [-p BEHAVIOR_POLICY_PATH] [-s SEEDS [SEEDS ...]] [-c CPUS] [--dataset_sizes DATASET_SIZES [DATASET_SIZES ...]] [--n_wedges N_WEDGES [N_WEDGES ...]] [--target_ks TARGET_KS [TARGET_KS ...]] [--output_path OUTPUT_PATH] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  -p BEHAVIOR_POLICY_PATH, --behavior_policy_path BEHAVIOR_POLICY_PATH
                        file containing the behavior policy
  -s SEEDS [SEEDS ...], --seeds SEEDS [SEEDS ...]
                        list of seeds
  -c CPUS, --cpus CPUS  number of cpus to par
  --dataset_sizes DATASET_SIZES [DATASET_SIZES ...]
                        number of worker processes
  --n_wedges N_WEDGES [N_WEDGES ...]
                        list of hyper-parameters to be evaluated
  --target_ks TARGET_KS [TARGET_KS ...]
                        list of history sizes to be used by the target policy
  --output_path OUTPUT_PATH
                        output directory (default=results)
  --verbose             runs on verbose mode
```
