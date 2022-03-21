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
from scipy import ndimage
from hausdorff import hausdorff_distance

#-----------------------------------------------------#
#        Calculate : Simple Hausdorff Distance        #
#-----------------------------------------------------#
"""
References:
    A. A. Taha and A. Hanbury.
    "An Efficient Algorithm for Calculating the Exact Hausdorff Distance".
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. 37, no. 11, pp. 2153-2163, 1 Nov. 2015, doi: 10.1109/TPAMI.2015.2408351.

Implementation: https://github.com/mavillan/py-hausdorff
"""
def calc_SimpleHausdorffDistance(truth, pred, c=1, **kwargs):
    # Compute simple Hausdorff Distance
    hd = hausdorff_distance(truth, pred, distance="euclidean")
    # Return Hausdorff Distance
    return np.float64(hd)

#-----------------------------------------------------#
#       Calculate : Average Hausdorff Distance        #
#-----------------------------------------------------#
"""
References:
    Adel Kermi, Issam Mahmoudi, Mohamed Tarek Khadir. 2019.
    Deep Convolutional Neural Networks Using U-Net for Automatic Brain Tumor Segmentation in Multimodal MRI Volumes.
    https://link.springer.com/chapter/10.1007/978-3-030-11726-9_4

Implementation: https://github.com/Issam28/Brain-tumor-segmentation
"""
def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0], order=0)
    east = ndimage.shift(binary_map, [1, 0], order=0)
    north = ndimage.shift(binary_map, [0, 1], order=0)
    south = ndimage.shift(binary_map, [0, -1], order=0)
    cumulative = west + east + north + south
    border = ((cumulative < 4) * binary_map) == 1
    return border

def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg

def calc_AverageHausdorffDistance(truth, pred, c=1, **kwargs):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    # Obtain sets with associated class
    ref = np.equal(truth, c)
    seg = np.equal(pred, c)
    # Compute AHD
    ref_border_dist, seg_border_dist = border_distance(ref, seg)
    hausdorff_distance = np.max([np.max(ref_border_dist),
                                 np.max(seg_border_dist)])
    # Return AHD
    return hausdorff_distance
