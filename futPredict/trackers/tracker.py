#creates tracker for smart assigning of bouding boxes to a single object over many frames
# will use features such as trajectory, and clothing color
from ultralytics import YOLO 
import supervision as sv 
import numpy as np
import pandas as pd 
import pickle 
import os 
import sys
import cv2
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path): # constructor taking a 'model_path' argument that specifies the path to pre-trained YOLO model weights
        self.model = YOLO(model_path) # creates YOLO model from path and stores it into model attrubute
        self.tracker = sv.ByteTrack()
    
    def add_position_to_tracks(self, tracks):
        for object, object_tacks in tracks.items():
            for frame_num, track in enumerate(object_tacks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball': 
                        position = get_center_of_bbox(bbox)
                    else: 
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position





    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions] # convert to pandas data frame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # interpolate 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # interpolate first frame if it is the missing one

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames): 
        batch_size = 20 # how many frames to process at a time 
        detections = [] # list storing all detections (object predictions) made by YOLO model 
        for i in range(0, len(frames), batch_size): # batch processing. i begins at zero with a step size of batch_size (20), up until end of list of frames is reached
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1) # predicts the next 20 frames staring at the i'th frame that
                                                                                      # only returning objects that have a 10% confidence score or greater
            detections += detections_batch 
        return detections # accumulated detection results   
    #detect objects in a series of video frames, and convert to supervision fromat (sv.Detections.from_ultralytics)
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None): #read_from_stub = false controls wether function runs from scratch (False means it will), and the stub_path

        if read_from_stub == True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames) # process the frames in batches, run detection on them, and returns the results in detections
        
        # Initializes an empty dictionary, tracks, with lists for storing the object tracks for players, referees, and the ball.
        tracks={
            "players" : [],
            "referees" : [], 
            "ball" : []
        }


        for frame_num, detection in enumerate(detections): # iterate through each detection, and it's frame number (detections holds results for each frame. loop will process each frames detections one by one)
            cls_names = detection.names # is a dictionary mapping class IDs to class names (e.g., {0: 'person', 1: 'car', ...}).
            cls_names_inv = {v:k for k, v in cls_names.items()} # inverts keys, and class names {'person': 0, 'car': 1, ...} 

            #convert to supervision detection format 
            detection_supervision = sv.Detections.from_ultralytics(detection) #convert to supervision format 

            # convert goalkeeper to normal player obj 
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # track objects: The ByteTrack algorithm updates object tracks across frames 
                            # based on the detections in the current frame.
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # For each frame, the method initializes empty dictionaries for players, referees, and ball in the tracks dictionary. 
            # Each dictionary will store track IDs (identifiers for tracked objects) and their bounding boxes for that frame.
            tracks["players"].append({}) #.append({}) appends an empty dictionary {} to the "players" list within the tracks dictionary (this is done for each frame_num)
            tracks["referees"].append({}) 
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks: # loop over each detection with track
                bbox = frame_detection[0].tolist() # extract bouding box (frame_detection[0] because in output the first list of info is bounding boxes and so on). tolist() used to convert bbox tensor format to list format
                cls_id = frame_detection[3] # extract class of object (player, referee, ball)
                track_id = frame_detection[4] # extract unique ID to track same object across frames

                if cls_id == cls_names_inv['player']:
                    #list index 
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox} # tracks["players"] accesses "players" list in tracks dictionary, tracks["players"][frame_num] accesses dictionary {} for current frame, tracks["players"][frame_num][track_id] Adds a new entry to the dictionary for the current frame

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox" : bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox" : bbox}

        if stub_path is not None: # if there is no stub_path we save it using pickle 
            with open(stub_path,'wb') as f: # 'wb' write binary, pickle stores data in binary format 
                pickle.dump(tracks, f) # serializes the Python object tracks (which in this case is a dictionary) and writes it to the file f.
                # "Serialization" means converting a Python object (like a dictionary, list, or any other object) into a byte stream, 
                # which can then be saved to a file or transmitted over a network.
    
        return tracks 

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # takes the bottom of bounding box y value, and casts into int 
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)), angle=0.0, startAngle=-45, endAngle=235, color=color,thickness=2, lineType=cv2.LINE_4)

        #draw rectangle for player number 
        rectangle_width = 40
        rectangle_height = 20
        #using top left, and bottom right of original bounding box
        x1_rect = x_center - rectangle_width// 2
        x2_rect = x_center + rectangle_width// 2
        y1_rect = (y2 - rectangle_height//2) + 15 # + 15 buffer 
        y2_rect = (y2 + rectangle_height//2) + 15 

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
        
            #write player numbers in rectangle 

            #where it will be written 
            x1_text = x1_rect + 12 
            if track_id > 99:
                x1_text -= 10 
        
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1]) #bottom point 
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x,y], [x - 10, y - 20], [x + 10, y - 20]]) # create three points 
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame 
              
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw semi transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of times each team has the ball 
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100: .2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100: .2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control): #replaces box bounding boxes with circles
        output_video_frames = [] # end result after desired circles 
        for frame_num, frame in enumerate(video_frames): # cycle through each frame 
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #draw player circles 
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,225)) # if no team color is found, red
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False): # inverted triangle on player who has ball
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 225))

            #draw referee circles 
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 225))
            
           # draw ball 
            for track_id, ball in ball_dict.items(): 
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # draw team ball control 
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)

        return output_video_frames
    
