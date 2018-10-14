"""
Estimates internal camera parameters using cv2 and a checkerboard pattern
"""
import pickle

import numpy as np
import cv2


def sample_video_file(file_name, no_target_frames=50):
    """
    Reads a video file, uniformly samples frames and returns them in a list
    """
    cap = cv2.VideoCapture(file_name)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    no_frame_step = no_frames / no_target_frames

    frames = []
    for ind in range(no_target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind * no_frame_step)
        _, frame = cap.read()
        frames.append(frame)

    return frames


def generate_3d_points(pattern_size, square_size=3):
    """
    Generates 3D points corresponding to pattern corners. square_size is in cms
    """
    object_points = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    object_points = object_points * square_size
    return object_points


def interal_calibration(frames, pattern_size=(8, 6)):
    """
    Estimates internal camera parameters using the provided frame list
    """
    object_points = generate_3d_points(pattern_size)
    object_point_list = []
    image_point_list = []
    for frame in frames:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, image_points = cv2.findChessboardCorners(gray_image, pattern_size)
        if ret:
            object_point_list.append(object_points)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_points_refined = cv2.cornerSubPix(gray_image, image_points, (11, 11), (-1, -1),
                                                    criteria)
            image_point_list.append(image_points_refined)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(object_point_list, image_point_list,
                                               gray_image.shape[::-1], None, None)
    print 'RMS reprojection error:', ret
    if ret > 1:
        print 'Error too high! (>1)'

    return mtx, dist


def main():
    """
    Reads 'int_calib.avi', uniformly samples frames and estimates internal camera parameters
    """
    frames = sample_video_file('int_calib.avi')
    mtx, dist = interal_calibration(frames)

    with open('int_calib.p', 'wb') as file_calib:
        calib_dict = {'K': mtx, 'dist': dist}
        pickle.dump(calib_dict, file_calib, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
