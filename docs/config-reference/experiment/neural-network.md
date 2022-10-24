# Neural network

The neural network model is an abstract model that can be used to create any
neural network model.

It is then up to the user to specify the architecture of the neural network
model.

While the `args` in the `model` section vary according to the model, the
`args` in the `training` section are the same for all neural network models.

```yaml
training:
    enabled: true
    args:
        epochs: 50
        optimizer:
            name: SGD
            args:
                lr: 0.0005
        loss:
            name: BCEWithLogitsLoss
        callbacks:
          - name: callback1
            args:
                arg1: value1
                arg2: value2
                ...
          - ...
```

The `epochs` argument specifies the number of epochs to train the model for.

The `optimizer` argument specifies the optimizer to use. It has two keys:
`name`, which contains the name of the optimizer, and `args`, which contains
the arguments that will be passed when creating the optimizer.

The `loss` argument specifies the loss function to use. It has two keys:
`name`, which contains the name of the loss function, and `args`, which
contains the arguments that will be passed when creating the loss function.

The `callbacks` argument specifies the callbacks to use. It is a list of
callbacks, each of which has a `name` and an optional `args` field.

## Available optimizers

There are some optimizers already available in Revelio:

* SGD
* Adam

In both optimizers the `args` field allows the same fields as the
corresponding PyTorch optimizer; therefore, at least the `lr` field must be
specified.

## Available loss functions

There are some loss functions already available in Revelio:

* BCEWithLogitsLoss
* BCELoss

In both loss functions the `args` field allows the same fields as the
corresponding PyTorch loss function.

## Available callbacks

There are some callbacks already available in Revelio:

* `EarlyStopping`
* `ModelCheckpoint`
* `TensorBoard`

The details of each callback are described in the following sections.

### EarlyStopping

The `EarlyStopping` callback stops the training when a monitored quantity has
stopped improving.

```yaml
callbacks:
  - name: EarlyStopping
    args:
        monitor: val_loss
        min_delta: 0.001
        patience: 5
        direction: min
        restore_best_weights: true
```

The `EarlyStopping` callback has the following arguments:

* `monitor` (optional string, default: `val_loss`): the quantity to be monitored.
    It can be either `loss` or any of the metrics specified in the `metrics` section
    of the configuration file. If you want to monitor a validation-time metric,
    prepend the metric name with `val_`.
* `min_delta` (optional float, default: `0.0`): the minimum change in the monitored
    quantity to qualify as an improvement, i.e. an absolute change of less than
    `min_delta`, will count as no improvement.
* `patience` (optional int, default: `0`): the number of epochs with no improvement
    after which training will be stopped.
* `direction` (optional string, default: `min`): whether to monitor the quantity
    in an increasing or decreasing way. It can be either `min` or `max`.
* `restore_best_weights` (optional bool, default: `false`): whether to restore
    model weights from the epoch with the best value of the monitored quantity.
    If `false`, the model weights obtained at the last step of training are used.

### ModelCheckpoint

The `ModelCheckpoint` callback saves the model after every epoch.

```yaml
callbacks:
  - name: ModelCheckpoint
    args:
        file_path: /path/to/checkpoint.pt
        monitor: val_loss
        min_delta: 0.001
        direction: min
        save_best_only: true
```

The `ModelCheckpoint` callback has the following arguments:

* `file_path` (string): the path to the file where the model will be saved.
* `monitor` (optional string, default: `val_loss`): the quantity to be monitored.
    It can be either `loss` or any of the metrics specified in the `metrics` section
    of the configuration file. If you want to monitor a validation-time metric,
    prepend the metric name with `val_`.
* `min_delta` (optional float, default: `0.0`): the minimum change in the monitored
    quantity to qualify as an improvement, i.e. an absolute change of less than
    `min_delta`, will count as no improvement.
* `direction` (optional string, default: `min`): whether to monitor the quantity
    in an increasing or decreasing way. It can be either `min` or `max`.
* `save_best_only` (optional bool, default: `false`): whether to save the model
    only if the monitored quantity has improved.

The `file_path` argument can contain the following placeholders, wrapped around curly
braces:

* `{epoch}`: the epoch number.
* `{now}`: the current date and time.
* `{metric}`: the value of the monitored metric (e.g. `{val_loss:.4f}`).
    Formatting can be specified after the metric name, as in the example.
    For more information on formatting, see the
    [Format String Syntax on the Python documentation](https://docs.python.org/3/library/string.html#formatstrings).

### TensorBoard

The `TensorBoard` callback logs the training metrics to TensorBoard.

```yaml
callbacks:
  - name: TensorBoard
    args:
        log_dir: /path/to/logs
        profile: false
```

The `TensorBoard` callback has the following arguments:

* `log_dir` (string): the path to the directory where the logs will be saved.
* `profile` (optional bool, default: `False`): whether to profile the training
    process.

The TensorBoard callback reports all the metrics specified in the `metrics`
section of the configuration file, as well as the loss.

For each metric, the callback reports the value of the metric at the end of
each epoch (with the `epoch_` prefix),
as well as the value of the metric at the end of each batch,
both for training and validation.

For example, if the `metrics` section of the configuration file contains
`accuracy`, the callback will report the following metrics:

* `epoch_accuracy`
* `epoch_val_accuracy`
* `accuracy`
* `val_accuracy`

Also, at the very first batch of the first epoch, the callback reports
the images in the training and validation set, as well as the corresponding labels.

*[SGD]: Stochastic Gradient Descent
*[BCE]: Binary Cross Entropy
