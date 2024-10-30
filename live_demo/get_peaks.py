import pandas as pd
from peakdetect import peakdetect
import numpy as np

def calculate_avg_angles(angles):
    avg_angles = ((angles.back_angles+angles.back_angles2)/2) + ((angles.legs_angles+angles.legs_angles2)/2) + ((angles.elbow_angles+angles.elbow_angles2)/2)
    return(avg_angles)

def states(avg_angles):
    peaks = peakdetect(avg_angles, lookahead=5, delta=10)
    high_peaks = pd.DataFrame(peaks[0])
    high_peaks.columns = ["consec","value"]
    #high_peaks = high_peaks.loc[high_peaks.value > np.mean(avg_angles)]
    high_peaks = high_peaks.loc[high_peaks.value > np.quantile(avg_angles, .75)]                              
    high_peaks["type"] = "peak"
    low_peaks = pd.DataFrame(peaks[1])
    low_peaks.columns = ["consec","value"]
    #low_peaks = low_peaks.loc[low_peaks.value < np.mean(avg_angles)]
    low_peaks = low_peaks.loc[low_peaks.value < np.quantile(avg_angles, .15)]
    low_peaks["type"] = "valley"
    final_peaks = pd.concat([high_peaks,low_peaks])
    return(final_peaks)