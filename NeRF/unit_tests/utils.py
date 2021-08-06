from math import acos, atan2, cos, pi, sin
from numpy import array, cross, dot, float64, hypot, zeros

#### Print Utils #####

tol = 1e-4

def print_separator():
    print("\n")
    for i in range(2):
        print("#"* 50)
    print("\n")

def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    
    From https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py#L178
    Copyright (C) Edward d'Auvergne
    """

    # Axes.
    axis = zeros(3, float64)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = hypot(axis[0], hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta