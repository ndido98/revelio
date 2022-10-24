# Metrics

There are several metrics already available in Revelio:

* Accuracy
* True positive rate
* True negative rate
* BPCER@APCER
* EER

No metric has any arguments, except for the following:

## BPCER@APCER

The BPCER@APCER metric has the following arguments:

* `thresholds` (required list of floats): the thresholds to use to compute the BPCER@APCER metric.
    The most used thresholds are 0.1, 0.05, 0.01 and 0.001.

*[BPCER@APCER]: Bona Fide Presentation Classification Error Rate at a given Attack Presentation Classification Error Rate
*[EER]: Equal Error Rate
