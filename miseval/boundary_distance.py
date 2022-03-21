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
from skimage.measure import find_contours
import dictances

#-----------------------------------------------------#
#            Calculate : Boundary Distance            #
#-----------------------------------------------------#
"""
Computes distance of segmentation boundaries between ground truth and prediction.

List of available distances:
    Bhattacharyya distance 	            bhattacharyya
    Bhattacharyya coefficient           bhattacharyya_coefficient
    Canberra distance 	                canberra
    Chebyshev distance 	                chebyshev
    Chi Square distance 	            chi_square
    Cosine Distance 	                cosine
    Euclidean distance 	                euclidean
    Hamming distance 	                hamming
    Jensen-Shannon divergence 	        jensen_shannon
    Kullback-Leibler divergence         kullback_leibler
    Mean absolute error 	            mae
    Taxicab geometry 	                manhattan, cityblock, total_variation
    Minkowski distance 	                minkowsky
    Mean squared error 	                mse
    Pearson's distance 	                pearson
    Squared deviations from the mean 	squared_variation

Distance Pooling (how to combine computed distances to a single value):
    Distance Sum                        sum
    Distance Averaging                  mean
    Minimum Distance                    amin
    Maximum Distance                    amax

Distance Implementation:
    Credits to Luca Cappelletti
    https://github.com/LucaCappelletti94/dictances

"""
def calc_Boundary_Distance(truth, pred, c=1, distance="euclidean",
                           pooling="mean", **kwargs):
    # Obtain sets with associated class
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    # Initialize result list
    res_dist = []
    try:
        # Compute boundary map for gt & pd
        gt_bm = np.concatenate(find_contours(gt))
        pd_bm = np.concatenate(find_contours(pd))
        # Compute distance between each coord
        for i in range(gt_bm.shape[0]):
            for j in range(pd_bm.shape[0]):
                # Create dictionary as interface for dictances package
                a = {"x": gt_bm[i][0],
                     "y": gt_bm[i][1]}
                b = {"x": pd_bm[j][0],
                     "y": pd_bm[j][1]}
                # Compute distance via dictances
                dist = getattr(dictances, distance)(a, b)
                # Append to result list
                res_dist.append(dist)
    except Exception as e:
        print("Distance", distance, "does NOT support all edge cases:", e)
    # Apply pooling function
    res = getattr(np, pooling)(res_dist)
    # Return AUC
    return res
