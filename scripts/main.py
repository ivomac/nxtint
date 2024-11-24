"""Main script for training a model."""

##############
### IMPORT ###
##############

import torch

from nxtint import Config, ModelID, Sequence, SequenceTransformer, Trainer

Config.init_dirs()

print(f"Used:\n{ModelID.used()}\nSuggested:\n{ModelID.new()}")

##################
### PARAMETERS ###
##################

model_id = "gentle-octopus"

config_override = SequenceTransformer.load_config(model_id)

#################
### SEQUENCES ###
#################

lin = [
    Sequence.linear(6, 6, [2]),
    Sequence.linear(3, 3, [2, 1]),
    Sequence.linear(20, 10, [1, 1]),
    Sequence.linear(3, 3, [3, 2, 1]),
    Sequence.linear(6, 6, [2], shift=10),
]

coup = [
    Sequence.coupled([1, 1], [1, 1], [[1, 1], [0, 2]]),
    Sequence.coupled([2, 2], [2, 2], [[1, 2], [2, 1]]),
    Sequence.coupled([4, 4], [4, 4], [[2, 2], [2, 2]]),
    Sequence.coupled([20, 20], [10, 10], [[1, 1], [1, 1]]),
    Sequence.coupled([6, 6], [4, 4], [[2, 3], [3, 2]], shift=10),
]

###################
### START MODEL ###
###################

with Config.override(**config_override):
    model = SequenceTransformer(model_id=model_id)

################
### TRAINING ###
################

with Config.override(**config_override):
    Trainer.train_from_phases(
        model,
        [
            lin[:1],
            lin[:2],
            lin[:3],
            lin[2:],
            [*lin[2:], *coup[:1]],
            [*lin[2:], *coup[:2]],
            [*lin[2:], *coup[2:3]],
            [*lin[3:], *coup[2:4]],
            [lin[3], *coup[3:]],
        ],
    )

###############
### TESTING ###
###############

x, y = next(coup[2])

y_pred = model(x).predict()

for i in range(len(x)):
    x_str = " ".join([str(xi) for xi in x[i].tolist()])
    print(f"{x_str} -> {y_pred[i].item()} ({y[i].item()} expected)")
    print()

######################
### MANUAL TESTING ###
######################

x = torch.as_tensor([[1, 3, 7, 13, 15, 19, 23, 27]])
y = torch.as_tensor([0])

y_pred = model(x).predict()

for i in range(1):
    x_str = " ".join([str(xi) for xi in x[0].tolist()])
    print(f"{x_str} -> {y_pred[i].item()} ({y[i]} expected)")
    print()

##################
### SAVE MODEL ###
##################

with Config.override(**config_override):
    model.save()

###########
### END ###
###########
