import cv2

from shape_transfer_utils import BodyShapeTransfer
from CDCL.inference_skeletons import init_model
from CDCL.inference_skeletons import get_control_points_new

def draw_keypoints(img, control_points):
    for part, partname in zip(control_points.values(), control_points.keys()):
        # print(partname)
        if(partname=="control_arm_l"):
            point_colour = (255, 0, 255)
        elif(partname=="control_arm_r"):
            point_colour = (0, 255, 255)
        elif(partname=="control_leg_l"):
            point_colour = (255, 255, 0)
        elif(partname=="control_leg_r"):
            point_colour = (0, 0, 255)
        elif(partname=="control_body"):
            point_colour = (255, 0, 0)
        else:
            point_colour = (255, 255, 255)
        for pt in part:
            if(pt[0] >= 0 and pt[1] >= 0):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, point_colour, -1)
    return img

def show_keypoint_img(image_path, control_points):
    # print(control_points)
    img = cv2.imread(image_path)
    img = draw_keypoints(img, control_points)
    cv2.imshow("test",img)
    cv2.waitKey(0)

def save_keypoint_img(image_path, control_points, save_path):
    img = cv2.imread(image_path)
    img = draw_keypoints(img, control_points)
    cv2.imwrite(save_path,img)

def save_segs_img(image_path, segs, save_path):
    img = cv2.imread(image_path)
    fusion = cv2.add(img, segs)
    cv2.imwrite(save_path,fusion)

def human_body_transform(image_path, params, model, model_params):
    # control_points, segs = get_control_points(image_path)
    control_points, segs = get_control_points_new(image_path, params, model, model_params)

    # cv2.imshow("test", segs)
    # cv2.waitKey(0)
    
    # show_keypoint_img(image_path, control_points)
    # save_keypoint_img(image_path, control, "D:/user/Desktop/test.jpg")

    # test_transfer = BodyShapeTransfer(control, {'height': 170, 'BMI': 21, 'type': 'X'}, {'height': 170, 'BMI': 21, 'type': 'X'})
    # dstImg = test_transfer.transform(image_path)
    # cv2.imshow("test", dstImg)

    return control_points, segs


if __name__ == "__main__":
    params, model, model_params = init_model()

    for i in range(1, 42):
        image_path = "C:/Users/26271/Desktop/2202_HumanBodyTransfer/Human-Body-Transfer/data/imgs/"+ str(i) +".jpg"
        keypoint_path = "C:/Users/26271/Desktop/2202_HumanBodyTransfer/Human-Body-Transfer/data/keypoints/"+ str(i) +".jpg"
        segs_path = "C:/Users/26271/Desktop/2202_HumanBodyTransfer/Human-Body-Transfer/data/segs/"+ str(i) +".jpg"
        control_points, segs = human_body_transform(image_path, params, model, model_params)
        save_keypoint_img(image_path, control_points, keypoint_path)
        save_segs_img(image_path, segs, segs_path)
