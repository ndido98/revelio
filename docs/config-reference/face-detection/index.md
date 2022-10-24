# Face detection

The face detection section of the configuration file contains the settings that
are used to detect faces in the images of the dataset, if needed.

```yaml
face_detection:
    enabled: true
    output_path: /path/to/face_detection_output
    algorithm:
        name: mtcnn_detector
        args:
            arg1: value1
            arg2: value2
            ...
```

The `enabled` field is a boolean that indicates whether face detection should be
performed or not.

The `output_path` field is a string that indicates the path where the face
detection output should be saved.

The `algorithm` field has two subfields: `name` specifies the name of the face detector
to use, and `args` is a dictionary of arguments that will be passed to it.

!!! note
    The `name` field is case-insensitive, and all hyphens and underscores are ignored,
    so `MTCNN_DETECTOR`, `MtcnnDetector`, `mtcnn-detector` and `mtcnn_detector` are all
    equivalent and will use the same `MTCNNDetector`.

## Face detectors

There are three face detectors already available in Revelio:
* [dlib](http://dlib.net/)
* [OpenCV](https://opencv.org/)
* [MTCNN](https://github.com/timesler/facenet-pytorch)

Both **dlib** and **MTCNN** also extract facial landmarks, while **OpenCV** does not.

### dlib

The `dlib` face detector is based on the [dlib](http://dlib.net/) library.

The `dlib` face detector has the following arguments:

* `landmark_predictor_path` (optional path, default: `None`): the path to the file
    containing the facial landmark predictor model.
    If `None`, facial landmarks will not be extracted.
    The landmark predictor model can be downloaded from
    [the dlib website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

```yaml
face_detection:
    enabled: true
    output_path: /path/to/face_detection_output
    algorithm:
        name: dlib_detector
        args:
            landmark_predictor_path: /path/to/landmark_predictor.dat
```

### OpenCV

The `opencv` face detector is based on the [OpenCV](https://opencv.org/) library.

The `opencv` face detector has the following arguments:

* `classifier_path` (required path): the path to the file containing the OpenCV
    face detector model that the `CascadeClassifier` will load.
    The most common model is `haarcascade_frontalface_default.xml`, which is
    available on the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

```yaml
face_detection:
    enabled: true
    output_path: /path/to/face_detection_output
    algorithm:
        name: opencv_detector
        args:
            classifier_path: /path/to/classifier.xml
```

### MTCNN

The `mtcnn` face detector is based on the [MTCNN](https://github.com/timesler/facenet-pytorch)
library.

The `mtcnn` face detector has no arguments.

```yaml
face_detection:
    enabled: true
    output_path: /path/to/face_detection_output
    algorithm:
        name: mtcnn_detector
```
