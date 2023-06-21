import os
import time
import dlib
import imutils
import datetime
import cv2 as cv 
import numpy as np
import pygame as pg
from threading import Thread 
from imutils import face_utils
from imutils.video import VideoStream	
from scipy.spatial import distance as dist

def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear_value = (A + B) / (C * 2)
    
    return ear_value 

def play_sound():
    sound = pg.mixer.Sound("alarm.wav")
    sound.play(-1)
    time.sleep(2.5)
    sound.stop()
    
pg.mixer.init()

# EAR_THRESHOLD = 0.18
# EAR_THRESHOLD = 0.20
EAR_THRESHOLD = 0.225
# EAR_THRESHOLD = 0.25
EYE_CLOSED_MIN_FRAME = 10
EYE_CLOSED_COUNTER = 0
ALARM_ON = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

vs=VideoStream(usePiCamera=True).start()
vs.camera.rotation = -90
time.sleep(1.0)

# ---------------------------------------- remove the files older than 3 days ------------------------------------
current_datetime = datetime.datetime.now()
threshold = datetime.timedelta(days=7)
directory = "data/"  
files = os.listdir(directory)

for file_name in files:
    file_path = os.path.join(directory, file_name)
    
    if os.path.isfile(file_path):
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        time_difference = current_datetime - modified_time
        
        if time_difference > threshold:
            # Remove the file
            os.remove(file_path)
            print(f"File '{file_name}' removed successfully.")
            print("File older than 7 days have been removed completely.")
# ---------------------------------------------------------------------------------------------------

while 1:
    frame_count = 0
    start_time = time.time()
    
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gray_frames = detector(gray)
    for gray_frame in gray_frames:
        face = predictor(gray, gray_frame)
        face = face_utils.shape_to_np(face)
        
        left_eye = face[42:48]
        right_eye = face[36:42]
        
        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)

        ear = (left_EAR + right_EAR) / 2.0

        left_eye_hull = cv.convexHull(left_eye)
        right_eye_hull = cv.convexHull(right_eye)
        
        cv.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv.putText(frame, f"FPS: {round(fps, 2)}", (10, 280), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if ear < EAR_THRESHOLD:
            EYE_CLOSED_COUNTER += 1
            
            if EYE_CLOSED_COUNTER >= EYE_CLOSED_MIN_FRAME:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=play_sound)
                    t.start()
                cv.putText(frame, "Sleepiness Detected!", (190, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                current_date = datetime.datetime.now()
                formatted_date = current_date.strftime("%Y-%m-%d")

                file_name = f"data/{formatted_date}.txt"
                current_time = current_date.strftime("%H:%M:%S")
                if os.path.exists(file_name):
                    with open(file_name, 'a') as file:
                        file.write(f"{ear:.2f} \t\t Sleepiness Detected \t\t {current_time}\n")
                else:
                    with open(file_name, 'w') as file:
                        file.write(f"Date: {formatted_date}\n\n{'-'*50}\n\n")
                        file.write(f"EAR Value \t Sleepiness Status \t\t Time\n\n")
                        file.write(f"{ear:.2f} \t\t Sleepiness Detected \t\t {current_time}\n")

                print(f"Text file '{file_name}' created and written successfully.")
                
            else:
                ALARM_ON = False
        else:
            EYE_CLOSED_COUNTER = 0
            ALARM_ON = False
            
        cv.putText(frame, "EAR: {:.2f}".format(ear), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(frame, f"Eyes Closed Frame: {EYE_CLOSED_COUNTER}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv.imshow("Detecting Driver's Face...", frame)
    key = cv.waitKey(1) & 0xFF

cv.destroyAllWindows()
vs.stop()