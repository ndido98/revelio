# Preprocessing

The preprocessing section of the configuration file contains the settings that are used
to preprocess the images in the dataset, right before passing them to the model.

```yaml
preprocessing:
  steps:
    - uses: step1
      args:
        arg1: value1
        arg2: value2
        ...
    - ...
```

The `steps` list contains all the preprocessing steps that will be applied to the images
in the dataset.

Each step has two keys: `uses`, which contains the name of the step, and `args`, which
contains the arguments that will be passed to the step.

!!! note
    The `uses` field is case-insensitive, and all hyphens and underscores are ignored,
    so `STEP1`, `Step1`, `step-1` and `step_1` are all equivalent and will use the same
    `Step1` step.

The transformations are applied in the order they are specified in the
configuration file.

## Preprocessing steps

There are some preprocessing steps already available in Revelio.

### Resize

The `resize` step resizes the images to the specified size.

The `resize` step has the following arguments:

* `width` (required int): the width of the resized images.
* `height` (required int): the height of the resized images.
* `algorithm` (optional string, default: `cubic`): the algorithm to use to resize the images.
    The available algorithms are the same as the ones available in OpenCV:

    * `nearest`: nearest-neighbor interpolation
    * `linear`: bilinear interpolation
    * `cubic`: bicubic interpolation (default)
    * `area`: resampling using pixel area relation
    * `lanczos4`: Lanczos interpolation over 8x8 neighborhood

* `keep_aspect_ratio` (optional bool, default: `True`): whether to keep the aspect ratio
    of the images when resizing them.
* `fill_mode` (optional string, default: `constant`): the strategy used for filling in
    newly created pixels, which can appear when using the `keep_aspect_ratio` option.
    The available strategies are the same as the ones available in OpenCV:

    * `constant`: the pixels are filled with black (default, e.g. `000000|abcdefgh|000000`)
    * `reflect`: the pixels are filled with the reflection of the image (e.g. `gfedcb|abcdefgh|gfedcba`)
    * `replicate`: the pixels are filled with the last pixel of the image (e.g. `aaaaaa|abcdefgh|hhhhhhh`)
    * `wrap`: the pixels are filled with the wrap of the image (e.g. `cdefgh|abcdefgh|abcdefg`)

### To float

The `to_float` step converts the images to floating point images in the range `[0, 1]`.

The `to_float` step has no arguments.

### Normalize

The `normalize` step normalizes the images using the specified mean and standard deviation.

The `normalize` step has the following arguments:

* `mean` (optional list of floats, default: `None`):
    the mean to use to normalize the images, one element per BGR channel.
* `std` (optional list of floats, default: `None`):
    the standard deviation to use to normalize the images, one element per BGR channel.
* `preset` (optional string, default: `None`): the preset to use to normalize the images.
    The available presets are:

    * `imagenet`: the mean and standard deviation used to normalize the images in the
        ImageNet dataset.

Either `preset` or both `mean` and `std` must be specified.

!!! warning
    Watch out for the order of the channels in the mean and standard deviation values.
    The order is BGR (i.e. OpenCV convention), not RGB.
