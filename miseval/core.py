#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External modules
import numpy as np
# Internal modules
from miseval import metric_dict

#-----------------------------------------------------#
#             Core Hub Function: Evaluate             #
#-----------------------------------------------------#
""" blabla. todo

Metric Parameter:
    The miseval eavluate function can be run with different metrics as backbone.
    You can pass the following options to the metric parameter:
    - String naming one of the metric labels, for example "DSC"
    - Directly passing a metric function, for example calc_DSC (from dice.py)
    - Passing a custom metric function

    List of metrics : See miseval/__init__.py under section "Access Functions to Metric Functions"

    The classes in a segmentation mask must be ongoing starting from 0 (integers from 0 to n_classes-1).

    A segmentation mask is allowed to have either no channel axis or just 1 (e.g. 512x512x1),
    which contains the annotation.
    The only exception from this is by activating the probabilities parameter, which results
    that a segmentation mask must have the same number of channels as n_classes (e.g. 512x512x8 if n_classes==8).
    Also the probabilities have to be in range between 0 to 1 and sum up to 1 (softmax).

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
             probabilities=False):
    # Initialize metric function
    if isinstance(metric, str) and metric in metric_dict:
        eval_metric = metric_dict[metric]
    else : eval_metric = metric

    # if multi_class == false and classes==2
    # run only on class==1

    # else iterate with for loop over it

   ...
   if multi_class : return results
   else : return results[1]
