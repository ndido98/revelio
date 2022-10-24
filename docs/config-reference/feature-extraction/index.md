# Feature extraction

The feature extraction section of the configuration file contains the settings that
are used to extract features from the images in the dataset.

```yaml
feature_extraction:
    enabled: false
    output_path: /path/to/feature_extraction_output
    algorithms:
      - name: feature_extractor
        args:
            arg1: value1
            arg2: value2
            ...
```

The `enabled` field is a boolean that indicates whether feature extraction should be
performed or not.

The `output_path` field is a string that indicates the path where the feature extraction
output should be saved.

The `algorithms` list contains all the algorithms that will be used to extract features
of different types from the images in the dataset.

Each algorithm has two keys: `name`, which contains the name of the algorithm, and
`args`, which contains the arguments that will be passed to the algorithm.

!!! note
    The `name` field is case-insensitive, and all hyphens and underscores are ignored,
    so `FEATURE_EXTRACTOR`, `FeatureExtractor`, `feature-extractor` and `feature_extractor`
    are all equivalent and will use the same `FeatureExtractor` algorithm.
