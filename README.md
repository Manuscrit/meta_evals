# Representation Engineering

Experiments with representation engineering. There's been a bunch of recent work ([1](https://arxiv.org/abs/2310.01405), [2](https://arxiv.org/abs/2308.10248), [3](https://arxiv.org/abs/2212.03827)) into using a neural network's latent representations to control & interpret models.

This repository contains utilities for running experiments (the `repeng` package) and a bunch of experiments (the notebooks in `experiments`).

## Installation
```bash
git clone git@github.com:Manuscrit/meta_evals.git
cd meta_evals
pip install -e .
# Or if using poetry [DEPRECATED]
poetry install 
```


Also need to install Anthropic's model written evals
```bash
# Outside of the meta_evals directory
git clone git@github.com:anthropics/evals.git
# Then change the constant PERSONA_REPO in meta_evals/meta_evals/utils/constants.py to the path of the model-written evals directory
```

## Reproducing experiments

### How well do truth probes generalise?
[Report](https://docs.google.com/document/d/1tz-JulAUz3SOc8Qm8MLwE9TohX8gSZlXQ2Y4PBwfJ1U).

1. Install the repository, as described [above](#installation).
2. Create a dataset of activations: `python experiments/gather_activation_dataset.py`.
    - This will upload the experiments to S3. Some tinkering may be required to change the upload location - sorry about that!
3. Run the analysis: `python experiments/plot_comparisons.py`.
    - This will write plots to `./data/plots/comparison`.

This is split into two scripts as only the first requires a GPU for LLM inference.
