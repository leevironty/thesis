# Leevi's MSc Thesis

In progress.


## Environment setup

- New Python virtual environment.
- ```pip install poetry```
- ```poetry install```
- ```pre-commit install```




## Idea

Solving TimPass is hard, but PESP is easier. The edge weights affect the PESP solution quality, as we want to reroute the passengers after setting the timetable (essentially, a sequential problem). If we could know the TimPass solution weights beforehand, we could use PESP and a shortest path algorithm to obtain a solution to TimPass.

From this, arises the idea of predicting the TimPass edge weights.


## The Question

Can we find good solutions to TimPass efficiently, by first predicting the optimal edge weights and then solving the resulting PESP problem?


## Planned approach
- If not shown already, show that if we can predict the weights correctly, we would obtain an optimal solution to TimPass.
- Generate training data with TimPass
    - Take a medium-sized problem/problems from TimPassLib
    - Take only a subset of the OD pairs to obtain a new, fast-to-solve problem.
    - Maybe also randomly drop some lines/stops/transfer options, so that we have multiple line structures with small differences in the training data.
    - Randomize bounds too
    - Generate many problem instances and the optimal TimPass solutions to those instances in this way.
- Experiment with multiple ML models
    - The objective here is to find something, that can predict the weights with some reasonable accuracy.
    - Evaluate prediction accuracy.
    - Possible approaches:
        - Predict the passenger counts directly
        - Predict the optimal route as a sequence -> calculate the passenger counts.
        - Predict the optimal timetable/edge duration -> calculate the shortest paths -> passenger counts.
    - Possible models to try:
        - Graph NN / something with attention mechanisms
        - Transformers / seq2seq for generating the paths between OD pairs.
        - Something simple, like SVR w/ nonlinear kernels / tree-based methods
- Run PESP with the generated predictions
    - Evaluate PESP + SP solutions, and compare them with known best solutions & run times.
    - Evaluate how the fitted models generalize to unseen networks.
        - Could we briefly fine-tune the model by generating a few examples in the same manner as in the training data generation? And could we do this within the time limit of 1h?




## Possible issues to think about beforehand
- Coming up with the design matrix for the "simple" models could be non-trivial. Likewise, not immediately clear how the NN-based approaches would be implemented.
- Small changes in one part of the EAN could lead to large changes in distant edge weights. How can we take this into account in the choice of the prediction model & training data generation?
- How well could we even predict the weights? What would be the mechanism that could allow us to do so?
- Could there be multiple solutions to TimPass? If yes, could then predicting a single set of weights be difficult?
- How much in common do near-optimal solutions to TimPass have? If not much, could this help us by allowing small errors in the weight prediction while still converging to the optimal solution?

## Additional ideas
- If we could have the model predict a distribution instead of a point value, maybe we could then sample those distributions and try multiple predictions for the weights. Maybe this could help, as the good solutions may look very different.
    - We need to sample the distributions jointly, could this be tricky? Depends on the used model?
- Could we just generate the training network graphs randomly? Could this bring more variation and thus better generalization?
