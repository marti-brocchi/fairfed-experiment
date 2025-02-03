# FairFed experiment with Declearn
## Initialization
To setup the environment:
- Clone the repository
- Create a new virtual environment and activate it
- Install required packages

```
git clone git@github.com:marti-brocchi/fairfed-experiment.git
cd fairfed-experiment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiment
The experiment simulates a federated learning setup with 5 clients with the same data distribution (IID), created using the `no_cov_same_size` setup.

The experiment consists of 10 rounds. All the clients participate in each round (there is no client selection).
Evaluation is conducted at the end of each round, and the final metrics are collected through aggregation of clients' metrics.