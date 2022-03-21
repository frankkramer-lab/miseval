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
#              Calculate : Cross-Entropy              #
#-----------------------------------------------------#
""" Compute cross-entropy between truth and prediction probabilities.

    In information theory, the cross-entropy between two probability distributions p and q
    over the same underlying set of events measures the average number of bits needed to
    identify an event drawn from the set if a coding scheme used for the set is optimized
    for an estimated probability distribution q, rather than the true distribution p.

    Source: https://en.wikipedia.org/wiki/Cross_entropy

Pooling (how to combine computed cross-entropy to a single value):
    Distance Sum                        sum
    Distance Averaging                  mean
    Minimum Distance                    amin
    Maximum Distance                    amax
"""
def calc_CrossEntropy(truth, pred_prob, c=1, pooling="mean", provided_prob=True,
                      **kwargs):
    # Obtain binary classification
    if provided_prob : prob = np.take(pred_prob, c, axis=-1)
    else : prob = np.equal(pred_prob, c)
    gt = np.equal(truth, c).astype(int)
    # Add epsilon to probability to avoid zero divisions for log()
    prob = prob + np.finfo(np.float32).eps
    # Compute cross-entropy
    cross_entropy = - gt * np.log(prob)
    # Apply pooling function across all pixel classifications
    res = getattr(np, pooling)(cross_entropy)
    # Return Cross-Entropy
    return res
