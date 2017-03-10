# Inverse Dynamics Simulation for Learning Experiments

This repository contains a very simple but easy to extend simulation 
which enables us to run various inverse dynamics learning experiments,
and compare different approaches.

All following commands are assumed to be executed from the idsim root directory.

# Installation

For ease of use an installation script is provided which
sets up a virtual environement. We install all 
necessary dependencies into this virtual environment.

The user has the option to either use tensorflow CPU or
GPU. Note, one of the two versions has to be selected.

```bash
bash scripts/install.sh
```

# Iros Experiments

In order to run the Iros experiments please execute the following 
script:

```bash
bash scripts/exp_iros/create_experiment.sh
```

This will create a folder structure in logs.

```bash
ls logs/exp_iros/
```
Should now show a lot of folders with descriptive names for every single experiment.
Every single experiment folder should contain the following files
```bash
config.yaml       # All open parameters used for this experiment.
experiment.sh     # The script to run exactly this experiment.
results.sh        # The script to generate the data required to aggregate the results.
template_params.yaml  # The template parameters used to generate the config.yaml.
visualize.sh      # This script will visualize how the system evolves over time, and a lot of other properties ofr this experiment.
```

Most notably there are 4 files prefixed with cluster_.
The two files called cluster_exp.sub and cluster_res.sub can be used in a htcondor cluster environment.
This will run all experiments in parallel.

However, experiments can be run manually and sequential by using the scripts cluster_exp.sh and cluster_res.sh.

```bash
# The script cluster_exp.sh takes a single parameter, namely the index of the experiment.
# In this particular cases there exist 1280 experiments, thus, every single one can be run
# as follows.
# experiment 0;
bash logs/exp_iros/cluster_exp.sh 0 
# experiment 1;
bash logs/exp_iros/cluster_exp.sh 1

# A simple sequential execution could look like this;
for i in $(seq 1 1280);do
  bash logs/exp_iros/cluster_exp.sh $i
  bash logs/exp_iros/cluster_res.sh $i
done
```
In order to create the plots from the paper we have to aggregate all results.
```bash
source vpy/bin/activate
python inverse_dynamics/results.py \
  --dp_exp_root logs/exp_iros/ \
  --dp_output logs/exp_iros/results \
  --plot \
  --average_params '[NOISE_FORWARD_FRICTION, NOISE_FORWARD_STICTION, NOISE_SENSING, RANDOM_SEED]' \
  --q \
  --tau_feedback_applied  
```
The resulting plots will be located in logs/exp_iros/results.

