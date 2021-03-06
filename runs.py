# This file contains all configs for all of our approaches
# None of the below configs are directly used in code so it's safe to add/remove/edit any parts as you like.
# The only case one would want to use this is entering one of the keys as model_name in train.py
# So that the models and logs are saved under the directory with that specified name.

runs = {
    "1": {
        "name": "",
        "ssl_tasks": ["puzzle", "rotation"],
        "optimizer": {"name": "adabelief",
                      "lr": 0.0001,
                      "decay": 0.},
        "total_loss": "sum",
        "total_loss_weights": [1, ..., 1],
        "losses": {"emotion": "categorical_crossentropy",
                   "puzzle": "categorical_crossentropy",
                   "rotation": "categorical_crossentropy"},
        "ssl_losses_aggregation": True,
        "ssl_weights_auto_freeze": False},

    "2": {"name": "",
          "ssl_tasks": ["puzzle"],
          "optimizer": {"name": "adabelief",
                        "lr": 0.0001,
                        "decay": 0.},
          "total_loss": "sum",
          "total_loss_weights": [1, 1, 1],
          "losses": {"emotion": "categorical_crossentropy",
                     "puzzle": "categorical_crossentropy"},
          "ssl_losses_aggregation": True,
          "ssl_weights_auto_freeze": False},
    "3": {"name": "",
          "ssl_tasks": ["puzzle"],
          "optimizer": {"name": "adabelief",
                        "lr": 0.0001,
                        "decay": 0.},
          "total_loss": "geometric",
          "total_loss_weights": [1, 1, 1],
          "losses": {"emotion": "categorical_crossentropy",
                     "puzzle": "categorical_crossentropy"},
          "ssl_losses_aggregation": True,
          "geometric_focused_loss": False,
          "ssl_weights_auto_freeze": False},
    "4": {"name": "",
          "ssl_tasks": ["puzzle"],
          "optimizer": {"name": "adabelief",
                        "lr": 0.0001,
                        "decay": 0.},
          "total_loss": "geometric",
          "total_loss_weights": [1, 1, 1],
          "losses": {"emotion": "categorical_crossentropy",
                     "puzzle": "categorical_crossentropy"},
          "ssl_losses_aggregation": True,
          "geometric_focused_loss": True,
          "ssl_weights_auto_freeze": False},
    "5": {"name": "",
          "ssl_tasks": ["puzzle"],
          "optimizer": {"name": "adabelief",
                        "lr": 0.0001,
                        "decay": 0.},
          "total_loss": "geometric",
          "total_loss_weights": [30, 1, 1],
          "losses": {"emotion": "categorical_crossentropy",
                     "puzzle": "categorical_crossentropy"},
          "ssl_losses_aggregation": True,
          "geometric_focused_loss": False,
          "ssl_weights_auto_freeze": False}

}
