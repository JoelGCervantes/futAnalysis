# transforms dimensions of field in video taken from an angle, and transforms them into 
# a rectagnle where accurate calculations can be made
import numpy as np
import cv2


class viewTransformer():
    def __init__(self):
        field_width = 68
        field_length = 23.32

        self.pixel_vertices = np.array([ # dimensions of trapezoid 
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        self.target_vertices = np.array([ # transformed trapezoid 
            [0, field_width],
            [0, 0],
            [field_length, 0],
            [field_length,  field_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1,2) 
    
    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None: 
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = transformed_position