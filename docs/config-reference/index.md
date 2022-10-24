# Configuration file reference

Revelio is a declarative framework for running Morphing Detection Attack experiments,
therefore its configuration must be done using a [YAML](https://yaml.org/) file.

This section of the documentation contains all the allowed settings and their
description.

## Main structure

The main components of a configuration file are the following:

* seed (optional)
* datasets
* face detection
* augmentation
* feature extraction
* preprocessing
* experiment

The order of the components is not relevant, but it is recommended to follow the
order above, as it is the order in which the components are executed.
