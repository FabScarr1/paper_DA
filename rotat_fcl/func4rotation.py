# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:23:38 2024

@author: f_sca
https://math.stackexchange.com/questions/3175854/finding-perpendicular-vector-to-an-arbitrary-n-dimensional-vector
"""
import numpy as np
import math 
import torch
from torch import linalg as LA

def perpendicular_vector(v0):
    idx_max = torch.argmax(torch.abs(v0))

    v1 = torch.zeros(v0.shape)
    v1[idx_max] = -v0[(idx_max+1) % len(v0)]/v0[idx_max]
    v1[(idx_max+1) % len(v0)] = 1
    
    return v1

def perpendicular_vector_1(v0):
    #idx_max = np.argmax(np.abs(v0))

    v1 = np.zeros(v0.shape)
    v1[0] = v0[1]
    v1[1] = -v0[0]
    
    return v1

def angle(vector1, vector2):
    # cos(theta) = v1 dot v2 / ||v1|| * ||v2||
    import numpy
    numerator = numpy.dot(vector1, vector2)
    denominator = numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2)
    x = numerator / denominator if denominator else 0
    return numpy.arccos(x)


import math
import numpy as np


def rotation_from_angle_and_plane(angle, vector1, vector2, abs_tolerance=1e-10):

    #vector1 = np.asarray(vector1, dtype=float)
    #vector2 = np.asarray(vector2, dtype=float)
    #angle = torch.tensor(angle).cuda()
    vector1_length = LA.vector_norm(vector1)
    if torch.isclose(vector1_length, torch.tensor(0.), atol=abs_tolerance):
        raise ValueError(
            'Given vector1 must have norm greater than zero within given numerical tolerance: {:.0e}'.format(abs_tolerance))

    vector2_length = LA.vector_norm(vector2)
    if torch.isclose(vector2_length, torch.tensor(0.), atol=abs_tolerance):
        raise ValueError(
            'Given vector2 must have norm greater than zero within given numerical tolerance: {:.0e}'.format(abs_tolerance))

    vector2 /= vector2_length
    dot_value = torch.dot(vector1, vector2)

    if torch.abs(dot_value / vector1_length ) > 1 - abs_tolerance:
        raise ValueError(
            'Given vectors are parallel within the given tolerance: {:.0e}'.format(abs_tolerance))

    if abs(dot_value / vector1_length ) > abs_tolerance:
        vector1 = vector1 - dot_value * vector2
        vector1 /= LA.vector_norm(vector1)
    else:
        vector1 /= vector1_length


    vectors = torch.vstack([vector1, vector2]).T
    vector1, vector2 = torch.linalg.qr(vectors)[0].T

    V = torch.outer(vector1, vector1) + torch.outer(vector2, vector2)
    W = torch.outer(vector1, vector2) - torch.outer(vector2, vector1)

    return torch.eye(len(vector1)).cuda() + (torch.cos(angle) - 1)*V - torch.sin(angle)*W
