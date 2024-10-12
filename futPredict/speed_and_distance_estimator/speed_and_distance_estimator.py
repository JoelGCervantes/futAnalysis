
import sys 
import cv2

sys.path.append('../')
from utils import measure_distance, get_foot_position
class speedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        # total distance dictionary 
        total_distance = {}

        # loops through onjects inside of tracks 
        for object, object_tracks in tracks.items():
            if object == 'ball' or object == 'referee': # only want to get speed for players 
                continue 
            number_of_frames = len(object_tracks) #amount of frames 
            for frame_num in range(0, number_of_frames, self.frame_window): #loops through each frame starting at 0 through the amount of frames, by step size of 5
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1) # last frame might be more or less than 5 

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]: # if the track id is in the first frame, but not the last, continue 
                        continue

                    # we need player to be in botht the first and last frame to calculate their speed 
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None: # if player is not in both frames
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_mph = speed_meters_per_second * 2.23

                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_mph
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" in tracks.items() or object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue 

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed: .2f} mph", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance: .2f} m", (position[0],position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            output_frames.append(frame)
        
        return output_frames
        





