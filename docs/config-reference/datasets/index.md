# Datasets

This section contains the definition of the datasets used in the experiment,
during the training and evaluation phases.

The most minimal dataset definition is done in the following way:

```yaml
datasets:
  - name: dataset_name
    path: /path/to/dataset/root
    split:
        train: 0.7
        val: 0.1
        test: 0.2  
  - ...
```

The `name` field is used to identify the dataset in the experiment, and the `path`
field is the path to the root directory of the dataset.

The `split` field is used to define the split of the dataset into training, validation
and test sets.
The three numbers (each between 0 and 1, both inclusive) represent the percentage of
the dataset that will be used for each of the three sets.

!!! note
    The sum of the three numbers must be at most 1.

    However, it's not required that the sum of the three numbers is *exactly* 1:
    for instance, if you don't want to insert the dataset in the validation set,
    you can set the `val` field to 0.
    The same applies to both `train` and `test`.

    This feature is particularly useful when you want to use only part of a dataset
    for testing, because having it in its entirety would unbalance the test set.

These three fields are required, but you can also add other fields to the dataset
definition, as described in the following sections.

## Dataset loading

The way in which a dataset is loaded is defined by the so-called **dataset loader**,
which is a class that scans the dataset root directory and returns a list of
items with their respective classes, which are then used to create the whole dataset.

By default, the name of the dataset also determines the dataset loader that will be
used to load the dataset.
For instance, if the dataset name is `morphdb`, the dataset loader that will be used
is `MorphDBLoader`, which is the default dataset loader for the MorphDB dataset.

!!! note
    The dataset name is case-insensitive, and all hyphens and underscores are ignored,
    so `MORPHDB`, `MorphDB`, `morph-db` and `morph_db` are all equivalent and will use
    the same `MorphDBLoader`.

If you want to know how to implement a dataset loader,
[see the reference for further details](/reference/revelio/dataset/loaders/loader).

Some loaders can also accept additional parameters, which can be specified in the
`loader.args` section of the dataset definition:

```yaml
datasets:
  - name: dataset_name
    path: /path/to/dataset/root
    split:
        train: 0.7
        val: 0.1
        test: 0.2
    loader:
        args:
            arg1: value1
            arg2: value2
            ...  
  - ...
```

The `args` field is a dictionary of arguments that will be passed to the dataset loader.

Sometimes, it is useful to explicitly specify the dataset loader to use, even if the
dataset name would normally imply a different loader.
This can be done by specifying the `loader.name` field in the dataset definition:

```yaml
datasets:
  - name: dataset_name
    path: /path/to/dataset/root
    split:
        train: 0.7
        val: 0.1
        test: 0.2
    loader:
        name: MyCustomLoader
        args:
            arg1: value1
            arg2: value2
            ...  
  - ...
```

!!! warning
    Unlike the `name` field, the `loader.name` field is case-sensitive, and an exact
    match is required.

## Testing groups

By default, the model is evaluated on the entirety of the test set.
Sometimes it is useful to evaluate the performance of a model only on a subset
of the test set, rather than on its entirety.

Therefore, you can assign a dataset to multiple testing groups, and the metrics will
be computed for each group separately.

To do so, you can specify the `testing_groups` field in the dataset definition:

```yaml
datasets:
  - name: dataset_name
    path: /path/to/dataset/root
    split:
        train: 0.7
        val: 0.1
        test: 0.2
    testing_groups:
        - group1
        - group2
        - ...
  - ...
```

!!! note
    The `testing_groups` field is ignored if the dataset is not in the test set.

The model will be evaluated on each of the groups separately, and on the whole test set.

!!! warning
    Make sure that each group contains at least one item for each class, otherwise
    some metrics such as EER and BPCER@APCER may not be computed.

*[EER]: Equal Error Rate
*[BPCER@APCER]: Bona Fide Presentation Classification Error Rate, given a fixed Attack Presentation Classification Error Rate
