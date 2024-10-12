import pickle 
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class cameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15, 15), # window that we are going to search 
            maxLevel = 2, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # either loop over number of times in stopping criteria, or loop 10 times and find nothing above quality score 
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0: 20] = 1 # takes top portion
        mask_features[:, 900: 1050] = 1 #takes bottom portion 

        self.features = dict(
            maxCorners = 100, 
            qualityLevel = 0.3, # higher quality = higher quality features, but less amount of them 
            minDistance = 3, # minimum distance between features is 3 pixels 
            blockSize = 7,
            mask = mask_features # block size is the search size of mask_features specifiying where to get freatures from
        ) 

    def add_adjust_posistions_to_tracks(self, tracks, camera_movement_per_frame):
        #loop over each of the objects 
        for object, object_tracks in tracks.items():
            #loop over each frame in the tracks
            for frame_num, track in enumerate(object_tracks):
                # for each frame loop over each track_id 
                for track_id, track_info in track.items():
                    position = track_info['position'] # get position 
                    # adjust position 
                    camera_movement = camera_movement_per_frame[frame_num]
                    # subract x_movement from camera_movement_x and y_movement from camera_movement_y 
                    postition_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # assign it 
                    tracks[object][frame_num][track_id]['position_adjusted'] = postition_adjusted


    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):
        # read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]] * len(frames) # first list is for frames and second is for x and y positions multiplied by length of frames 
        
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) # convert image into gray image to extract features. old_gray is previous frame
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features) # extract corner features. ** to expand dictionary into parameters 

        for frame_num in range(1, len(frames)): # loop over each frame, from 0 + 1st frame 
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params) # 

            #measure distance between old and new features and determine if there is camera movement or not 
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)): # to loop through any two lists I must zip them first
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance 
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(camera_movement, f)
        
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement: .2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement: .2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames



