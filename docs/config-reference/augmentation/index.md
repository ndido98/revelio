# Augmentation

You can specify the data augmentation pipeline in the `augmentation` section
of the configuration file.
The augmentation pipeline is a list of transformations that are probabilistically applied
to the images in the dataset.
The transformations are applied in the order they are specified in the configuration file.

An example of the most minimal non-empty augmentation pipeline is the following:

```yaml
augmentation:
    enabled: true
    steps:
      - uses: random_crop
        args:
            crop_size: 224
```

The `enabled` field is a boolean that indicates whether data augmentation should be
applied to the dataset.
If `enabled` is `false`, the augmentation pipeline is ignored and the dataset is not
augmented.

The `steps` field is a list of transformations that are applied to the dataset.
Each transformation is specified by a dictionary with at least the `uses` field, which
indicates the name of the transformation to use.

A step can also have an `args` field, which is a dictionary of arguments that will be
passed to the transformation.

!!! note
    The `uses` field is case-insensitive, and all hyphens and underscores are ignored,
    so `RANDOM_CROP`, `RandomCrop`, `random-crop` and `random_crop` are all equivalent
    and will use the same `RandomCrop` transformation.

There are some other fields that can be specified for each augmentation step:

* `probability` (optional float, default: `1.0`): the probability that the transformation
    will be applied to the image.
    If `probability` is `0.5`, the transformation will be applied to half of the images
    in the dataset.
* `applies_to` (optional list of strings or ints, default: `all`): the list which contains
    the indices of the images that the transformation will be applied to, relative to the
    dataset element (which can have more than one image).
    If `applies_to` is `all`, the transformation will be applied to all images in the
    dataset element.
    If `applies_to` is `[0, 2]`, the transformation will be applied to the first and third
    images in the dataset element.
    There are some special values for `applies_to`:

    * `all`: the transformation will be applied to all images in the dataset element (this
        is the default value); if this value is used, no other value can be specified.
    * `probe`: the transformation will be applied to the first image in the dataset element,
        which by convention is the probe image.
    * `live`: the transformation will be applied to the second image in the dataset element,
        which by convention is the live image.

!!! note
    The probability of applying a transformation to an image is on the whole step, not
    on each image in the dataset element.

    For instance, if the dataset element has 3 images, and the step has `probability: 0.5`
    and `applies_to: [0, 2]`, the transformation will be atomically applied to both images
    with probability `0.5`: either both images will be transformed, or none of them will.
