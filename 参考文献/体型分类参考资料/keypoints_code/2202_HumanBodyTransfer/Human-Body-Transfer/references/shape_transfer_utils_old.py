#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jian-Wei ZHANG
@date: 2017/8/8
@update: 2020/9/25
@update: 2021/7/14: simplify usage
"""

import numpy as np
import matplotlib.pyplot as plt
from mesh_utils import mls_affine_deformation, mls_similarity_deformation, mls_rigid_deformation
from PIL import Image

from CDCL.inference_skeletons import controls

def gen_pq(controls, bodypart, rate):
    keypoints = controls["keypoints"]
    #  print(keypoints)

    if(bodypart=="ARM_LEN"):
        if(len(controls['control_arm_upper_l'])==0 or len(controls['control_arm_upper_r'])==0 or len(controls['control_arm_lower_l'])==0 or len(controls['control_arm_lower_r'])==0):
            return [], [], controls

        q = np.zeros((0,2))
        p=np.vstack((np.vstack((np.vstack((controls['control_arm_upper_l'], controls['control_arm_upper_r'])), controls['control_arm_lower_l'])), controls['control_arm_lower_r']))
        part_list = ['control_arm_upper_l','control_arm_upper_r','control_arm_lower_l','control_arm_lower_r']
        adjust = np.array([0, 0])
        for part in part_list:
            c0 = controls[part][0]
            c1 = (controls[part][2] + controls[part][3]) / 2.0
            c2 = (controls[part][4] + controls[part][5]) / 2.0
            c3 = (controls[part][6] + controls[part][7]) / 2.0
            c4 = controls[part][1]
            d1 = controls[part][3] - controls[part][2]
            d2 = controls[part][5] - controls[part][4]
            d3 = controls[part][7] - controls[part][6]

            c1 = c0 + (c1 - c0) * rate
            c2 = c0 + (c2 - c0) * rate
            c3 = c0 + (c3 - c0) * rate
            c4 = c0 + (c4 - c0) * rate

            h1 = c1 + d1 / 2.0
            h2 = c2 + d2 / 2.0
            h3 = c3 + d3 / 2.0

            l1 = c1 - d1 / 2.0
            l2 = c2 - d2 / 2.0
            l3 = c3 - d3 / 2.0

            q_part = np.array([c0, c4, l1, h1, l2, h2, l3, h3]) + adjust
            if(part == 'control_arm_upper_l'):
                adjust = c4 - controls['control_arm_lower_l'][0]
            elif(part == 'control_arm_upper_r'):
                adjust = c4 - controls['control_arm_lower_r'][0]
            else:
                adjust = np.array([0, 0])
            q = np.vstack((q, q_part))

        keypoints = np.delete(keypoints, [3, 4, 6, 7], axis = 0)
        controls['control_arm_upper_l'] = q[0:8]
        controls['control_arm_upper_r'] = q[8:16]
        controls['control_arm_lower_l'] = q[16:24]
        controls['control_arm_lower_r'] = q[24:32]
        # return p.astype(int), q.astype(int)
    elif(bodypart=="ARM_WID"):
        if(len(controls['control_arm_upper_l'])==0 or len(controls['control_arm_upper_r'])==0 or len(controls['control_arm_lower_l'])==0 or len(controls['control_arm_lower_r'])==0):
            return [], [], controls

        q = np.zeros((0,2))
        p=np.vstack((np.vstack((np.vstack((controls['control_arm_upper_l'], controls['control_arm_upper_r'])), controls['control_arm_lower_l'])), controls['control_arm_lower_r']))
        part_list = ['control_arm_upper_l','control_arm_upper_r','control_arm_lower_l','control_arm_lower_r']
        adjust = np.array([0, 0])
        for part in part_list:
            c0 = controls[part][0]
            c1 = (controls[part][2] + controls[part][3]) / 2.0
            c2 = (controls[part][4] + controls[part][5]) / 2.0
            c3 = (controls[part][6] + controls[part][7]) / 2.0
            c4 = controls[part][1]
            d1 = controls[part][3] - controls[part][2]
            d2 = controls[part][5] - controls[part][4]
            d3 = controls[part][7] - controls[part][6]

            h1 = c1 + (d1 / 2.0)*rate
            h2 = c2 + (d2 / 2.0)*rate
            h3 = c3 + (d3 / 2.0)*rate

            l1 = c1 - (d1 / 2.0)*rate
            l2 = c2 - (d2 / 2.0)*rate
            l3 = c3 - (d3 / 2.0)*rate

            q_part = np.array([c0, c4, l1, h1, l2, h2, l3, h3])
            q = np.vstack((q, q_part))

        keypoints = np.delete(keypoints, [3, 4, 6, 7], axis = 0)
        controls['control_arm_upper_l'] = q[0:8]
        controls['control_arm_upper_r'] = q[8:16]
        controls['control_arm_lower_l'] = q[16:24]
        controls['control_arm_lower_r'] = q[24:32]
    elif(bodypart=="LEG_LEN"):
        if(len(controls['control_leg_upper_l'])==0 or len(controls['control_leg_upper_r'])==0 or len(controls['control_leg_lower_l'])==0 or len(controls['control_leg_lower_r'])==0):
            return [], [], controls

        q = np.zeros((0,2))
        p=np.vstack((np.vstack((np.vstack((controls['control_leg_upper_l'], controls['control_leg_upper_r'])), controls['control_leg_lower_l'])), controls['control_leg_lower_r']))
        part_list = ['control_leg_upper_l','control_leg_upper_r','control_leg_lower_l','control_leg_lower_r']
        adjust = np.array([0, 0])
        for part in part_list:
            c0 = controls[part][0]
            c1 = (controls[part][2] + controls[part][3]) / 2.0
            c2 = (controls[part][4] + controls[part][5]) / 2.0
            c3 = (controls[part][6] + controls[part][7]) / 2.0
            c4 = controls[part][1]
            d1 = controls[part][3] - controls[part][2]
            d2 = controls[part][5] - controls[part][4]
            d3 = controls[part][7] - controls[part][6]

            c1 = c0 + (c1 - c0) * rate
            c2 = c0 + (c2 - c0) * rate
            c3 = c0 + (c3 - c0) * rate
            c4 = c0 + (c4 - c0) * rate

            h1 = c1 + d1 / 2.0
            h2 = c2 + d2 / 2.0
            h3 = c3 + d3 / 2.0

            l1 = c1 - d1 / 2.0
            l2 = c2 - d2 / 2.0
            l3 = c3 - d3 / 2.0

            q_part = np.array([c0, c4, l1, h1, l2, h2, l3, h3]) + adjust
            if(part == 'control_leg_upper_l'):
                adjust = c4 - controls['control_leg_lower_l'][0]
            elif(part == 'control_leg_upper_r'):
                adjust = c4 - controls['control_leg_lower_r'][0]
            else:
                adjust = np.array([0, 0])
            q = np.vstack((q, q_part))

        keypoints = np.delete(keypoints, [9, 10, 12, 13], axis = 0)
        controls['control_leg_upper_l'] = q[0:8]
        controls['control_leg_upper_r'] = q[8:16]
        controls['control_leg_lower_l'] = q[16:24]
        controls['control_leg_lower_r'] = q[24:32]
    elif(bodypart=="LEG_WID"):
        if(len(controls['control_leg_upper_l'])==0 or len(controls['control_leg_upper_r'])==0 or len(controls['control_leg_lower_l'])==0 or len(controls['control_leg_lower_r'])==0):
            return [], [], controls

        q = np.zeros((0,2))
        p=np.vstack((np.vstack((np.vstack((controls['control_leg_upper_l'], controls['control_leg_upper_r'])), controls['control_leg_lower_l'])), controls['control_leg_lower_r']))
        part_list = ['control_leg_upper_l','control_leg_upper_r','control_leg_lower_l','control_leg_lower_r']
        adjust = np.array([0, 0])
        for part in part_list:
            c0 = controls[part][0]
            c1 = (controls[part][2] + controls[part][3]) / 2.0
            c2 = (controls[part][4] + controls[part][5]) / 2.0
            c3 = (controls[part][6] + controls[part][7]) / 2.0
            c4 = controls[part][1]
            d1 = controls[part][3] - controls[part][2]
            d2 = controls[part][5] - controls[part][4]
            d3 = controls[part][7] - controls[part][6]

            h1 = c1 + (d1 / 2.0)*rate
            h2 = c2 + (d2 / 2.0)*rate
            h3 = c3 + (d3 / 2.0)*rate

            l1 = c1 - (d1 / 2.0)*rate
            l2 = c2 - (d2 / 2.0)*rate
            l3 = c3 - (d3 / 2.0)*rate

            q_part = np.array([c0, c4, l1, h1, l2, h2, l3, h3])
            q = np.vstack((q, q_part))

        keypoints = np.delete(keypoints, [9, 10, 12, 13], axis = 0)
        controls['control_leg_upper_l'] = q[0:8]
        controls['control_leg_upper_r'] = q[8:16]
        controls['control_leg_lower_l'] = q[16:24]
        controls['control_leg_lower_r'] = q[24:32]
    # elif(bodypart=="CHEST"):
    #     p=
    #     q=
    #     return
    # elif(bodypart=="BELLY"):
    #     p=
    #     q=
    #     return
    else:
        return [], [], controls

    #print(keypoints)
    keypoints = np.delete(keypoints, np.where(keypoints<0), axis = 0)
   # print(keypoints)
    p = np.vstack((p, keypoints))
    q = np.vstack((q, keypoints))
   # print("p:")
   # print(p.astype(int))
  #  print("q:")
   # print(q.astype(int))

    return p.astype(int), q.astype(int), controls

def deform_bodyshape(image, p, q):
    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    aug = np.ones_like(image)
    aug[vx, vy] = image[tuple(rigid)]
    return aug


if __name__ == "__main__":
    image_path = "images/WeChat Image_20210901143926.png"
    #棕色男半身体
    #WeChat Image_20210901151143.png
    
    #O1CN01L7obfz1LzVFPNJKGV_!!2696351370.png
    #女士棕色
    
    #O1CN01K18Dnv1LzVGfWbaSl_!!2696351370.png
    #女士黄色短袖
    
    #O1CN01IiQPzm1LzVDHVZMVy_!!2696351370.png
    #女式连衣裙
    
    #1632817114(1).png
    #拎着包
    controls = controls(image_path)
    print(controls)

    #bodypart1 = "ARM_LEN"  
    bodypart1 = "ARM_WID"
    bodypart2 = "LEG_LEN"
    #bodypart2 = "LEG_WID"
    # bodypart = "CHEST"
    # bodypart = "BELLY"

    image = np.array(Image.open(image_path))
    rate = 1
    p, q, controls = gen_pq(controls, bodypart1, rate)
    aug1 = deform_bodyshape(image, p, q)
    p, q, controls = gen_pq(controls, bodypart2, rate)
    aug2 = deform_bodyshape(aug1, p, q)

    # Show the result
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(aug2)
    ax[1].set_title("Rigid Deformation - %s: %.1f %%" % ("ARM LEN + WID", (rate - 1) * 100))

    for x in ax.flat:
        x.axis("off")
    
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


# def human_body_transform(image_path):
    # p = np.array([[0, 0], [0, 517], [798, 0], [798, 517],
    #     [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
    #     [243, 225], [211, 244], [253, 244]
    # ])
    # q = np.array([[0, 0], [0, 517], [798, 0], [798, 517],
    #     [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
    #     [243, 225], [207, 238], [261, 237]
    # ])

    # img = np.array(Image.open(image_path))
    # height, width, _ = img.shape
    # gridX = np.arange(width, dtype=np.int16)
    # gridY = np.arange(height, dtype=np.int16)
    # vy, vx = np.meshgrid(gridX, gridY)

    # rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    # aug = np.ones_like(img)
    # aug[vx, vy] = img[tuple(rigid)]

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].imshow(img)
    # ax[0].set_title("Original Image")
    # ax[1].imshow(aug)
    # ax[1].set_title("Rigid Deformation")

    # for x in ax.flat:
    #     x.axis("off")
    
    # plt.tight_layout(w_pad=1.0, h_pad=1.0)
    # plt.show()