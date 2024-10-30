import cv2
import numpy as np
#from scipy.ndimage import gaussian_filter1d
import math
import mediapipe as mp
import threading
from peakdetect import peakdetect
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    v1 = { "x":a.x - b.x, "y":a.y - b.y, "z": a.z - b.z }
    v2 = { "x":c.x - b.x, "y":c.y - b.y,  "z":c.z - b.z}
    v1mag = math.sqrt(v1["x"] * v1["x"] + v1["y"] * v1["y"] + v1["z"] * v1["z"])
    v1norm = { "x":v1["x"] / v1mag, "y":v1["y"] / v1mag, "z":v1["z"] / v1mag}
    v2mag = math.sqrt(v2["x"] * v2["x"] + v2["y"] * v2["y"] + v2["z"] * v2["z"])
    v2norm = { "x":v2["x"] / v2mag, "y":v2["y"] / v2mag, "z":v2["z"] / v2mag}
    calc = (v1norm["x"] * v2norm["x"]) + (v1norm["y"] * v2norm["y"]) + (v1norm["z"]  * v2norm["z"]) 
    return math.degrees(math.acos(calc)) 

def calculate_slope(a,b,c,d):
    #a l_shoulder, b r_shoulder, c l_hip, d r_hip    
    center_shoulder = [((a.x+b.x)/2), ((a.y+b.y)/2), ((a.z+b.z)/2)]
    center_hip = [((c.x+d.x)/2), ((c.y+d.y)/2), ((c.z+d.z)/2)]
    center_vector = [center_shoulder[0]-center_hip[0], center_shoulder[1]-center_hip[1], center_shoulder[2]-center_hip[2]]
    standing = 1
    if ((abs(center_vector[0]) > abs(center_vector[1])) or (abs(center_vector[2]) > abs(center_vector[1]))):
       standing = 0
    return standing

    

def remove_outliers(an_array):
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = an_array[not_outlier]
    return no_outliers


#¿Posiblemente optimizable?
def landmarks_points(image):
  BG_COLOR = (192, 192, 192) # gray
  with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5) as pose:
      # Convert the BGR image to RGB before processing.
      results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      return results.pose_world_landmarks
  
def limitLandMarks(image1):
  '''
  This function finds the maximum and minimum landmarks along both the x and y axes in the input image.
  Args:
      image: The input image with a prominent person whose pose landmarks need to be detected.
      
  Returns:
      x_final: The total projection along the x-axis.
      y_final: The total projection along the y-axis.
  '''

  # Use MediaPipe Pose to detect pose landmarks in the input image.
  results = pose.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

  if results.pose_landmarks:  # Check if any landmarks are found.

    # Define dummy maximum and minimum values for the x and y axes.
    xmax = -100
    xmin = 100
    ymax = -100
    ymin = 100

    points_of_interest = [11,12,23,24,25,26,27,28,29,30,31,32] # Without arms and without head points
    for i in points_of_interest:
      # Get the x and y coordinates of the current landmark.
      x = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].x
      y = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i)].y

      # Update the maximum and minimum values for the x and y axes if necessary.
      if x >= xmax:
        xmax = x
        Landmark_xmax = mp_pose.PoseLandmark(i).name
      if x <= xmin:
        xmin = x
        Landmark_xmin = mp_pose.PoseLandmark(i).name
      if y >= ymax:
        ymax = y
        Landmark_ymax = mp_pose.PoseLandmark(i).name
      elif y <= ymin:
        ymin = y
        Landmark_ymin = mp_pose.PoseLandmark(i).name

      # Calculate the total projection along the x and y axes.
      x_final = abs(xmax - xmin)
      y_final = abs(ymax - ymin)

    # Print the maximum and minimum landmarks along both the x and y axes.
    print(f'The maximum landmark on the x-axis is: {xmax} in {Landmark_xmax}')
    print(f'The minimum landmark on the x-axis is: {xmin} in {Landmark_xmin}')
    print(f'The maximum landmark on the y-axis is: {ymax} in {Landmark_ymax}')
    print(f'The minimum landmark on the y-axis is: {ymin} in {Landmark_ymin}')

  # Return the total projection along the x and y axes.
  return x_final, y_final

def classifyPose(x_final, y_final):

    '''
    This function classifies poses depending upon x and y projection.
    Args:
        x_final: The total projection along the x-axis.
        y_final: The total projection along the y-axis.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
        Label_code: will be 0 for Lying down and 1 for Standing up

    '''

    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    Label_code = 0

    # Specify the color (Red) with which the label will be written on the image.
    #color = (255, 0, 0)

    # Check if x projection is greater than y projection
    if x_final > y_final:
        label = 'Lying down'
        Label_code = 0

    # Check if y projection is greater than x projection
    elif x_final < y_final:
        label = 'Standing up'
        Label_code = 1         

    #elif label != 'Unknown Pose':
       # Update the color (to green) with which the label will be written on the image.
    #    color = (0, 0, 255)  

    return label, Label_code

def generate_angles_output_with_time(cap:cv2.VideoCapture, ms=250):
    back_angles = []
    elbow_angles = []
    legs_angles = []
    back_angles2 = []
    elbow_angles2 = []
    legs_angles2 = []
    armpit_angles2 = []
    armpit_angles = []

    slope = []
    time = []
    coefficients = []
    orientations = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = cv2.CAP_PROP_POS_MSEC
    success = True
    counter = 0
    def process_frame(img, frame_time, counter):
        #if frame_time % ms <= fps:
        #print(counter,frame_time,frame_time % ms)
        coefficients.append(frame_time % ms)
        landmarks_p = landmarks_points(img)
        xmax = -100
        xmin = 100
        ymax = -100
        ymin = 100
        if landmarks_p is not None:
            points = landmarks_p.landmark    
            for i in range(32):
                # Get the x and y coordinates of the current landmark.
                x = points[i].x
                y = points[i].y

                # Update the maximum and minimum values for the x and y axes if necessary.
                if x >= xmax:
                    xmax = x
                if x <= xmin:
                    xmin = x
                if y >= ymax:
                    ymax = y
                elif y <= ymin:
                    ymin = y
                # Calculate the total projection along the x and y axes.
                x_final = abs(xmax - xmin)
                y_final = abs(ymax - ymin)   
            orientations.append(classifyPose(x_final, y_final)[1])
            back_angles.append(calculate_angle(points[12],points[24],points[26]))
            elbow_angles.append(calculate_angle(points[12],points[14],points[16]))
            armpit_angles.append(calculate_angle(points[14],points[12],points[24]))
            legs_angles.append(calculate_angle(points[24],points[26],points[28]))  
            back_angles2.append(calculate_angle(points[11],points[23],points[25]))
            elbow_angles2.append(calculate_angle(points[11],points[13],points[15]))
            armpit_angles2.append(calculate_angle(points[13],points[11],points[23]))
            legs_angles2.append(calculate_angle(points[23],points[25],points[27]))   
            slope.append(calculate_slope(points[12],points[11], points[24], points[23]))          
            time.append(frame_time)
    hilos = []
    while success:
        success, img = cap.read()
        if success:
            hilo = threading.Thread(target=process_frame, args=[img,frame_time, counter])
            hilos.append(hilo)
            hilo.start()
            frame_time = frame_time + 1000/fps
            counter = counter + 1
    #for hilo in hilos:
    #   hilo.start()
    for h in hilos:
        h.join()
    return  { "back_angles":back_angles, "elbow_angles":elbow_angles, "armpit_angles":armpit_angles, "legs_angles":legs_angles, "back_angles2":back_angles2, "elbow_angles2":elbow_angles2, "armpit_angles2":armpit_angles2, "legs_angles2":legs_angles2, "orientation" : orientations, "slope":slope, "time":time}

def get_video_frame(vidcap,frame):    
    success,image = vidcap.read()
    count = 0
    image = None
    while success:
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file    
        success,image = vidcap.read()  
        if frame == count: 
            success = False        
        count += 1
    return image


