from ultralytics import YOLO 

model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print('##########################')
for box in results[0].boxes:
    print(box)

 #bounding box: a box that surrounds an object that sowftware (in this case YOLO) can detect. What software can detect in our case is determinef by array of names. 
# xywh: tensor([[387.8123, 336.4704,  26.4911,  60.2360]])
 # - xywh notation represents a bounding box with a center point on the object plus a width and height
 # - xyxy notaion represesnts a bounding box with two points. One on top left and other at bottom right of an object. 
 #   These points can also be labeled as ((x1, y1), (x2,y2)).

#step ?: We now have object detection with a little more fine tuned guidlines. We can detect goalkeepers, referees, the ball, 
        # and we can exclude people outside of the game (coaches, nearby persons). This can be seem by comparing mp4 in runs/detect/predict 1 + 2, and runs/detect/predict3
    
    # We now want to create better annotations for objects detected in video

#step ?2: after creating video_utils.py, and __init__.py we created fucntions read_video, and save_video. we then tested them. 
        #with successful completion we get 'output_video.avi' in output_videos folder 
   
    # We will now write the detection and tracking using bite tracker 
        # Tracking: assigning the same object (in this case player/ball/ref) to a single bounding box over many frames 
            #before tracking due to the xy, xy nature of the bouding box. nothing tells us that the bouding box around one object 
            # is the same as the bouding box surrounding the same object in the next frame
            # A good tracker will also assing the correct bouding box throught all frames to a single object (Through predicting trajectory and visual features (clothing color ect...))

#step ?+3: we now have created a tracker class that will track objects assigning a bounding box to a single object instead of many boxes to one item
           # where errors could occur. 

        # we now want to ovveride the goalkeepers boudning box to simply be a player since there are no stats that need to be done on the keeper. This will 
         # also allow there to be less mistakes and confusion when the model detects the goalkeeper as a player 

    #up until now we can see the supervision format of video with onject detection with bouding boxed identifying how many objects in a frame 
    # also the class names, and ids are shown. We are still not using 'mask' or 'confidence'
        # from get_object_tracks we have now introduced a new array "tracker_id=array" into out output that numbers our bounding boxes (1-n). as bounding boxes move the numbers in array 
        # will also reorder even though they began in ascending order

# step #?+4: the tracker is finished. Now we can correctly have an id for each bounding box which will follow the obj(s) throught the given frames
            
            # We now want to put the output into a format that can be used easily. 
            # we will use a dictionary of lists for easier referencing 

# step #? + 5: We have written functions to tell the tracker how we want to store the data we are looking for. (dictionary of lists)
            # for each frame we will see a dictionary of object tracks who has defined in it lists of player, referee, and the ball
            # in each frame the postion of each player, referee, and ball will be appended to each list, and stored in the treaker object
            # the tracker object is then returned once the function is finised 

        # we now want to save tracks object returned by get_object_tracks as a pickle file to not have to wait for it to run everytime

# step #? + 6: at this point we have created the pickle file that has created the stub_path so that get_object_tracks won't completely run again 
                # the stub_path pickle file is opened, read, and stored in a tracks object that will then be returned

                # we now want to see and visualize our results 
                # we also want circles instead of boxes for bounding boxes 

# step #? + 7: We have now got rid of rectangle boudning box, and replaced it with ellipses (red=player, yellow=referee). 
                #output is in avi file. 
            
            # We now want to diplay player numbers (a track_id below players)
                #this is done int two steps. 1. drawing rectangle. 2. adding number

#  step #? + 8: At this point players have rectangle with track_id inside it, but there is some changes in number toward in of video
                # and numbers go past 99, even though we created if statement in draw_ellipse ????

                # we now want to draw inverted triangle over ball 

# step #? + 9: triangle has been drawn. 
                #now we want to assign players their team by color clustering, understand what color the user is wearing, and assign a team 
                
            # we first start by take any player and take their image and save it 
            #9.b segment color of cropped players tshirt 
            #9.c using matplotlib.pyplot as plt, and cv2 I took the cropped image taken from main.py 
                # and made it visualoizable in notebook. cv2 treated it, and converted it into rgb from bgr

                #now since color is on upper half of player body (thier shirt) we will take the upper half of image 
            #9.d top half of image has been taken. now we need to remove background, so color can be accurately taken 
            #9.e We now want to create two image clusters using kmeans to find the background + player shirt picture 
            #9.f We find the non player cluster "color" by finding how kmeans assigned it (ask chatgpt how it works)
            #9.g now that we have gotten the rgb color of one team. we want to assign it to the correct team
            #9.h we have assigned team colors in team_assigner.py in assign_team_color method, and we have also 
                # written a method to assign a player to their respective team 

                #now we can use the color in tracker instead of hard coding in a color 
            
            #9.i we update player circles according to team colors in tracker.py

                # figure out error
                
# step#? + 10: We have assigned colors to teams
             # we now want to improve ball tracking 
             # problem is that the ball is detected in a frame, then isn't for a few, then it is again 
             # we want to track the ball better by drawing a line between the first and second frame the ball is detected
             # dividing the the distance by 2, and adding bounding boxes to the missing frames

# step #? + 11: up until now we have made ball tracking better, and though there is little bit of lag, 
                # this is sufficient for our project. 

                # We now want to add an invertered triangle over the player with the ball using the same logic as before 
                # this is done by assigning the inverted triangle to the player whose foot is closest to the ball 
                # we dont want the ball to appear over anyone if there is no foot close to the ball 

# step #? + 12: # The player who has the ball is now being tracked correctly with inverted triangle. 
                # we now want to assign team ball control. the percentage of time that each time has the ball

                # Done in main + tracker
                # goalkeeper is being detected as opposite team. fixed in team_assigner.py  
                
                # we have now added the team ball control in main and tracker.py 
                # we have correctly assigned the team for each goalkeeper, and they are now being incorporated
                # in the ball control stats 

# step #? + 13: we now want to mease the movement of the camera. Camera movement causes bounding boxes to move even if the players are not movig 
                # we want to counteract the movement of the camera with player bbox movement 
                # so we don't exaggerate player movement. This gives us better data when calculating player movement
                # and disance covered.

                # we want to detect featrues. detect corners at top and bottom + other features in same area 
                # we want to detect how much each feature moved in each frame using optical flow

# step #? + 14: We have now added a camera movement estimator in camere_movement_estimate.py and displayed it on the screen.
                # still need to review code 
                 
# step #? + 15: we wantt to make player positons according to camera movement
                # we do this by getting player positions (add player movement to tracks)
                # then subtract camera movement 

                # need to go to camera movement and adjust the camera movement that we have just added to tracksn in main

# step #? + 16: at this point the player movement has been adjusted (camera movement has been subtracted from total player movement)
                
                # we now want to do perspective transformation (view transfomer) to know how much a player has moved in meters in real world 
                # Since pixels don't line up 1-1 with distance being coverd in meters we need to use "Perspective transfomation"
                #This is done by creating a rectangle that sets boundaries(width of field 68m), choosing a midpoint, and creating two rectagnles two fill
                # space from each sideline to midpoint equally 

                #half a football field length is taken (105m /2 = 52.5m), the width (68m), and the length is divided into 9 equal segments 
                # (52.5m / 9 = 5.833m)
                # given the width of field (68m) and field segment lengths (5.833m), given the points of trapezoid on field, we can create a transformed rectangle 
                # to accurately compute from   

##################################
# bugs/issues encountered up to this point: RuntimeWarning. incompatible libraries Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp')
# may cause crashed or deadlocks. <---- need to look into more info on why and how to workaround or fix

#When converting the player movement to meters in real world I needed to use the dimensions of field, but since camera is at an angle
# the pixels detected are not portpotional (not the same from one sideline to middle, and middle to other sideline)
# need to make computer understand that the pixels from one side is same distance as distance to other side, even if they have different values
# This was solved in step #? + 16
#syntax errors
######################################


# step #? + 17: now the field dimensions have been transformed for accurate calculations to be made 

                # final step is to actually calculate the speed of players
