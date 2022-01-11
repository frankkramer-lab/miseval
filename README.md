# miseval: a metric library for Medical Image Segmentation EVALuation

```
placeholder for badges
```

The open-source and free to use Python package miseval was developed to establish a standardized medical image segmentation evaluation procedure. We hope that our this will help improve evaluation quality, reproducibility, and comparability in future studies in the field of medical image segmentation.

#### Guideline on Evaluation Metrics for 	Medical Image Segmentation

1. Use DSC as main metric for validation and performance interpretation.
2. Use AHD for interpretation on point position sensitivity (contour) if needed.
3. Avoid any interpretations based on high pixel accuracy scores.
4. Provide next to DSC also IoU, Sensitivity, and Specificity for method comparability.
5. Provide sample visualizations, comparing the annotated and predicted segmentation, for visual evaluation as well as to avoid statical bias.
6. Avoid cherry-picking high-scoring samples.
7. Provide histograms or box plots showing the scoring distribution across the dataset.
8. For multi-class problems, provide metric computations for each class individually.
9. Avoid confirmation bias through macro-averaging classes which is pushing scores via background class inclusion.
10. Provide access to evaluation scripts and results with journal data services or third-party services like GitHub and Zenodo for easier reproducibility.

## Implemented Metrics

| Metric      | Index in miseval |
| ----------- | ----------- |
| Dice Similarity Index | "DSC", "Dice", "DiceSimilarityCoefficient" |
| Intersection-Over-Union | "IoU", "Jaccard", "IntersectionOverUnion" |
| True Positive | "TP", "TruePositive" |
| False Positive | "FP", "FalsePositive" |
| True Negative | "TN", "TrueNegative" |
| False Negative | "FN", "FalseNegative" |

## How to Use

#### Example

```python


```

#### Core function: Evaluate()

Every metric in miseval can be called via our core function `evaluate()`.

The miseval eavluate function can be run with different metrics as backbone.  
You can pass the following options to the metric parameter:
- String naming one of the metric labels, for example "DSC"
- Directly passing a metric function, for example calc_DSC (from dice.py)
- Passing a custom metric function

List of metrics : See `miseval/__init__.py` under section "Access Functions to Metric Functions"

The classes in a segmentation mask must be ongoing starting from 0 (integers from 0 to n_classes-1).

A segmentation mask is allowed to have either no channel axis or just 1 (e.g. 512x512x1),
which contains the annotation.  
The only exception from this is by activating the probabilities parameter, which results
that a segmentation mask must have the same number of channels as n_classes (e.g. 512x512x8 if n_classes==8).
Also the probabilities have to be in range between 0 to 1 and sum up to 1 (softmax).

```python
"""
Arguments:
    truth (NumPy Matrix):            Ground Truth segmentation mask.
    pred (NumPy Matrix):             Prediction segmentation mask.
    metric (String or Function):     Metric function. Either a function directly or encoded as String from miseval or a custom function.
    multi_class (Boolean):           Boolean parameter, if segmentation is a binary or multi-class problem. By default False -> Binary mode.
    n_classes (Integer):             Number of classes. By default 2 -> Binary
    probabilities (Boolean):         Boolean parameter, if predicted segmentation (pred) is encoded as softmax output.
                                     By default False -> normal class vector expected.

Output:
    score (Float) or scores (List of Float)

    The multi_class parameter defines the output of this function.
    If n_classes > 2, multi_class is automatically True.
    If multi_class == False & n_classes == 2, only a single score (float) is returned.
    If multi_class == True, multiple scores as a list are returned (for each class one score).
"""
def evaluate(truth, pred, metric, multi_class=False, n_classes=2,
             probabilities=False)
```

## Installation


- **Install miseval from PyPI (recommended):**

```sh
pip install miseval
```

- **Alternatively: install miseval from the GitHub source:**

First, clone miseval using git:

```sh
git clone https://github.com/frankkramer-lab/miseval
```

Then, go into the miseval folder and run the install command:

```sh
cd miseval
python setup.py install
```

## Author

Dominik Müller\
Email: dominik.mueller@informatik.uni-augsburg.de\
IT-Infrastructure for Translational Medical Research\
University Augsburg\
Bavaria, Germany

## How to cite / More information

Coming soon

```
coming soon
```

Thank you for citing our work.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.\
See the LICENSE.md file for license rights and limitations.
