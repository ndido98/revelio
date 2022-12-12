# Experiment

The experiment section of the configuration file contains the settings that
are used to run the experiment.

```yaml
experiment:
    batch_size: 64
    model:
        name: model-name
        args:
            arg1: value1
            arg2: value2
            ...
    training:
        enabled: true
        args:
            arg1: value1
            arg2: value2
            ...
    scores:
        bona_fide: /path/to/bona_fide_scores.txt
        morphed: /path/to/morphed_scores.txt
        metrics: /path/to/metrics.json
    metrics:
      - name: metric1
        args:
          arg1: value1
          arg2: value2
      - name: metric2
      - ...
```

While many of the settings vary according to the model used, there are some
settings that are common to all models.

The `batch_size` setting specifies the batch size to use when training the model.

The `model` setting specifies the model to use. It has two keys: `name`, which
contains the name of the model, and `args`, which contains the arguments that
will be passed when creating the model.

Inside the `model` setting you can specify the `checkpoint` field, which
contains the path to a checkpoint file. If this field is specified, the model
will be loaded from the checkpoint file instead of being created from scratch.

The `training` setting specifies the training settings. It has two keys:
`enabled`, which is a boolean that specifies whether to train the model, and
`args`, which contains the arguments that will be passed when training the
model.
If the `enabled` field is set to `false`, the model will not be trained, and
the model specified in the `model` setting will run only in inference mode.

Each model has its own set of training arguments. For instance, a neural network has
a much more complicated set of training arguments than a simple linear model.

The `scores` setting specifies the paths to the files containing the scores
for the bona fide and morphed images. The scores are the result of the model
evaluation on the test set images.
The `scores.metrics` field contains the path to a JSON file where the metrics
for each testing group will be saved. This field is optional.

The `metrics` setting specifies the metrics to use to evaluate the model. It
contains a list of metrics, each of which has a `name` and an optional `args` field.

## Available models

There are two main models already available in Revelio:
a neural network and a random guesser.

### Neural network

The configuration of a neural network is quite complex, and it is described in
the [Neural network](neural-network.md) page.

### Random guesser

The random guesser is a simple model that always returns a random score between 0 and 1.

The random guesser has no arguments.

## Available metrics

There are several metrics already available in Revelio:
a complete list can be found in the [Metrics](metrics.md) page.
