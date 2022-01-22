from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import linalg as LA


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts[0].size == 0 and norm_prev_pts[1].size):
        print('no prev points')
    elif (norm_curr_pts[0].size == 0 and norm_curr_pts[1].size):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts,norm_curr_pts=[None]*2,[None]*2
    for i in range(2):
        norm_prev_pts[i] = normalize(prev_container.traffic_light[i], focal, pp)
        norm_curr_pts[i] = normalize(curr_container.traffic_light[i], focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = [[]]*2
    corresponding_ind = [[]]*2
    valid_vec = [[]]*2
    for i in range(2):
        for p_curr in norm_curr_pts[i]:
            corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts[i], foe)
            Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
            valid = (Z > 0)
            if not valid:
                Z = 0
            valid_vec[i].append(valid)
            P = Z * np.array([p_curr[0], p_curr[1], 1])
            pts_3D[i].append((P[0], P[1], P[2]))
            corresponding_ind[i].append(corresponding_p_ind)
        pts_3D[i] = np.array(pts_3D[i])
    return corresponding_ind, np.array(pts_3D), valid_vec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return np.array([[(points[0] - pp[0]) / focal, (points[1] - pp[1]) / focal] for points in pts])


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return np.array([[(points[0] * focal) + pp[0], (points[1] * focal) + pp[1]] for points in pts])


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    if len(EM)>0:
        R = EM[:3, :3]
        t = EM[:3, 3]
    else:
        R = np.eye(3)
        t = [1, 1, 1]
    tZ = t[2]
    foe = [t[0] / tZ, t[1] / tZ]
    return R, foe, tZ


# def rotate(pts, R):
#     # rotate the points - pts using R
#     return [np.matmul(R, np.append(point, 1)) for point in pts]
#
#
# def find_corresponding_points(p, norm_pts_rot, foe):
#     # compute the epipolar line between p and foe
#     # run over all norm_pts_rot and find the one closest to the epipolar line
#     # return the closest point and its index
#     m = (foe[1] - p[1]) / (foe[0] - p[0])
#     n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
#     min_d, min_idx = float('inf'), -1
#     for index, point in enumerate(norm_pts_rot):
#         distance = abs((m * point[0] + n - point[1]) / sqrt(m ** 2 + 1))
#         if min_d > distance:
#             min_d, min_idx = distance, index
#     return min_idx, norm_pts_rot[min_idx]
#
#
# def calc_dist(p_curr, p_rot, foe, tZ):
#     # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
#     # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
#     # combine the two estimations and return estimated Z
#     x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
#     y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
#     x_dist = abs(p_curr[0] - p_rot[0])
#     y_dist = abs(p_curr[1] - p_rot[1])
#     return (abs(1 - x) * x_dist + abs(1 - y) * y_dist) / (x_dist + y_dist)

def rot(R, pt):
    return np.dot(R, np.array([pt[0], pt[1], 1]))

def rotate(pts, R):
    # rotate the points - pts using R
    arr_returned = [[]]*2
    for i in range(2):
        for pt in pts[i]:
            res = rot(R, pt)
            arr_returned[i].append([res[0] / res[2], res[1] / res[2]])
    return arr_returned

def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    distances = np.array([abs((m * pt[0] + n - pt[1]) / sqrt((m**2) + 1)) for pt in norm_pts_rot])
    return np.argmin(distances), norm_pts_rot[np.argmin(distances)]

def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    Zx = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    Zy = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])

    x_diff = abs(p_rot[0] - p_curr[0])
    y_diff = abs(p_rot[1] - p_curr[1])

    return (x_diff / (x_diff + y_diff)) * Zx + (y_diff / (x_diff + y_diff)) * Zy


