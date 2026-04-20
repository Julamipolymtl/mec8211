import os
import numpy as np 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

path_data = "C:/Users/janam/Downloads/vv"
all_data = {}

conversion = 0.00980665

for root, dirs, files in os.walk(path_data):
    for filename in sorted(files):
        if filename.endswith('.txt'):
            # immediate parent folder name
            subfolder_name = os.path.basename(root)
            file_path = os.path.join(root, filename)
            data = np.loadtxt(file_path, unpack=True)
            
            # dictionary with folder and filename as keys
            if subfolder_name not in all_data:
                all_data[subfolder_name] = {}
                
            # filename as the inner key
            file_key = os.path.splitext(filename)[0]
            all_data[subfolder_name][file_key] = data

config = "F"

srq = []
for filename, columns in all_data[config].items():
    disp = columns[1]
    force = columns[2]

    print(f"Processing: {filename}")

    srq_file = []
    # find peaks in displacement
    peaks, _ = find_peaks(disp)

    for i in peaks:
        # take only positive peaks
        if force[i] >= 10.0:
            srq_file.append(force[i])
            srq.append(force[i])
            print(f"displacement: {disp[i] - disp[0]}, force: {force[i]}")

    file_srq_avg = np.average(srq_file) * conversion
    print(f"SRQ, Specimen {filename}: {file_srq_avg} N")
    
srq_avg = np.average(srq) * conversion
print(f"SRQ : {srq_avg} N")

