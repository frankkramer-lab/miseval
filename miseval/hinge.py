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

#-----------------------------------------------------#
#                Calculate : Hinge loss               #
#-----------------------------------------------------#
""" Compute Hinge loss between truth and prediction probabilities.

    In machine learning, the hinge loss is a loss function used for training classifiers.
    The hinge loss is used for "maximum-margin" classification.

Pooling (how to combine computed Hinge losses to a single value):
    Distance Sum                        sum
    Distance Averaging                  mean
    Minimum Distance                    amin
    Maximum Distance                    amax
"""
def calc_Hinge(truth, pred_prob, c=1, pooling="mean", provided_prob=True,
               **kwargs):
    # Obtain binary classification
    if provided_prob : prob = np.take(pred_prob, c, axis=-1)
    else : prob = np.equal(pred_prob, c)
    gt = np.equal(truth, c).astype(int)
    # Convert ground truth 0/1 format to -1/+1 format
    gt = np.where(gt==0, -1, gt)
    # Compute Hinge
    hinge_total = np.maximum(1 - gt * prob, 0)
    # Apply pooling function across all pixel classifications
    hinge = getattr(np, pooling)(hinge_total)
    # Return Hinge
    return hinge
