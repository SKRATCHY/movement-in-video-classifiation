import pandas as pd
import cv2
from peakdetect import peakdetect
import numpy as np
import sys
from angles_detection import generate_angles_output_with_time
import get_peaks



def main(video, ejercicio):
    '''
    flag = False
    while not flag:
        valor = input("Variation (or NA), if multiple separate them with '-'\n")
        if len(valor) == 0:
            print("Please type the Variation")
        else:
            variation = valor
            flag = True

    flag = False
    while not flag:
        valor = input("Equipment (or NA), if multiple separate them with '-'\n")
        if len(valor) == 0:
            print("Please type the equipment")
        else:
            equipment = valor
            flag = True

    flag = False
    while not flag:
        valor = input("Primary Muscle, if multiple separate them with '-' (or NA)\n")
        if len(valor) == 0:
            print("Please type the Primary Muscle")
        else:
            primary_muscle = valor
            flag = True
        
    flag = False
    while not flag:
        valor = input("Assisting Muscle (or NA), if multiple separate them with '-'\n")
        if len(valor) == 0:
            print("Please type the Assisting Muscle")
        else:
            assisting_muscle = valor
            flag = True

    flag = False
    while not flag:
        valor = input("Exercise type (or NA), if multiple separate them with '-'\n")
        if len(valor) == 0:
            print("Please type the Exercise type")
        else:
            exercise_type = valor
            flag = True
    '''
    try:
        angles = pd.DataFrame(generate_angles_output_with_time(cv2.VideoCapture(video)))
    except:
        print("Video couldn't be read")
        exit()
    video_file_name = video.split("/")[-1].split(".")[-2]
    angles["avg_angles"] = get_peaks.calculate_avg_angles(angles)
    min_max = get_peaks.states(angles["avg_angles"])
    angles["consecutivo"] = range(angles.shape[0])
    to_train = angles.merge(min_max, how="inner", left_on="consecutivo", right_on="consec").fillna(0)
    to_train["name"] = ejercicio
    to_train["video_name"] = video_file_name
    #to_train["variation"] = variation
    #To_train["equipment"] = equipment
    #to_train["primary_muscle"] = primary_muscle
    #to_train["assisting_muscle"] = assisting_muscle
    #to_train["exercise_type"] = exercise_type
    training_data = pd.read_csv("exercise_train_data.csv")
    exercise_train_data = pd.concat([training_data,to_train[training_data.columns]])
    exercise_train_data.to_csv("exercise_train_data.csv", index=False)
    #to_train.to_csv("exercise_train_data.csv", index=False)



if __name__== "__main__":
    print(sys.argv)
    if len(sys.argv) < 3:
        print("Please send one video as parameter, and it's label")
        exit()
    main(sys.argv[1],sys.argv[2])