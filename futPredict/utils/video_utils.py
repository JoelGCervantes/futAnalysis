import cv2 # library to read in, and save the video 

def read_video(video_path): #function takes a video_path argument, and returns a list of frames 
    cap = cv2.VideoCapture(video_path) # video capture variable using cv2 method. 
    frames = [] 
    while True:
        ret, frame = cap.read() # .read() will return a flag (into ret). if flag is true it will also return a frame. 
        if not ret: #if flag is false function with break 
            break
        frames.append(frame) #otherwise frame will be appended to frames array, and loop will restart
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #define an output format 'XVID'
    #define a video writer that takes in video path (a string), output video type, frames per second, and frame (width, height)
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))   
    for frame in output_video_frames: #loop over each frame, and write the frame to video writer 
        out.write(frame)
    out.release()
