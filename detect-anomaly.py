import numpy as np
from sklearn.preprocessing import StandardScaler
from lib.anomaly import detect_anomalies_autoencoder, detect_anomalies_isolation_forest,\
    detect_anomalies_lof, detect_anomalies_one_class_svm, detect_anomalies_stdev
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import argparse
import os
from collections import Counter

def log_diff(arr : np.array) -> np.array:
    """Calculate log diff by first checking and correcting for non-positive elements"""
    # check if there are any non-positive values
    if np.any(arr <= 0):
        # if there are, add the absolute value of the minimum element plus one to every element
        arr += abs(np.min(arr)) + 1

    # now you can safely calculate the difference and logarithm
    logs = np.log(arr)
    return np.diff(logs)

def difference_twice(arr : np.array) -> np.array:
    return np.diff(np.diff(arr))

# create parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='Input the file path to a one-column csv file with no header with time-series values.')
parser.add_argument('--check_stationarity', type=bool, default=True)
parser.add_argument('--auto_stationarity', type=bool, default=True)
parser.add_argument('--anomaly_algorithm', nargs='+', default=["all"], choices=['all', 'autoencoder', 'isolation_forest', 'lof', 'stdev'], 
                    help='Choose one or more anomaly detection algorithms. Options are: "all", "autoencoder", "isolation_forest", "lof", "stdev"')
parser.add_argument('--threshold_autoencoder', type=float, default=2)
parser.add_argument('--contamination_isolation_forest', type=str, default="auto")
parser.add_argument('--contamination_lof', type=str, default="auto")
parser.add_argument('--threshold_stdev', type=float, default=2)

# parse arguments
args = parser.parse_args()

# parameters
file = args.file
check_stationarity = args.check_stationarity
auto_stationarity = args.auto_stationarity
anomaly_algorithm = args.anomaly_algorithm
threshold_autoencoder = args.threshold_autoencoder
contamination_isolation_forest = args.contamination_isolation_forest
contamination_lof = args.contamination_lof
threshold_stdev = args.threshold_stdev

# validate contamination parameters
if 'isolation_forest' in anomaly_algorithm and contamination_isolation_forest != "auto":
    try:
        contamination_isolation_forest = float(contamination_isolation_forest)
    except Exception as e:
        raise ValueError(f"contamination_isolation_forest must either be 'auto' or a float in the range (0,0.5]: {str(e)}")
    
if 'lof' in anomaly_algorithm and contamination_lof != "auto":
    try:
        contamination_lof = float(contamination_lof)
    except Exception as e:
        raise ValueError(f"contamination_lof must either be 'auto' or a float in the range (0,0.5]: {str(e)}")

# validate file
if not file.endswith(".csv"):
    raise ValueError("The input file must be a csv file.")

if not os.path.isfile(file):
    raise FileNotFoundError("Unable to find the input file.")

# load file
df = pd.read_csv(file)
df = df.dropna()

if len(df.columns) > 1:
    raise ValueError("The input file must have only one column.")

# convert to numeric
try:
    df = df.apply(pd.to_numeric)
except Exception as e:
    print(f"Failure converting values to numeric. Make sure every value is a number: {str(e)}")

# convert df to numpy array
data = df.to_numpy()

# normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 1))
data = data.squeeze()

# store offset from original data after applying stationarity methods (if applied)
offset_data = 0

# funcs to make time-series stationary and respective offset to original time-series
funcs_stationarity = [(np.diff,1,"Differencing"),
        (log_diff,1,"Log-differencing"),
        (difference_twice,2,"2-differencing"),
        (np.cbrt,0,"Cube root")]
stationarity_func_used = None # saves stationarity func used

# check if time-series is stationary
if check_stationarity:
    result_adf = adfuller(data)
    p_value = result_adf[1]

    # if auto_stationarity is off, print warning
    if not auto_stationarity:
        if p_value > 0.05:
            print(f"WARNING: the provided time-series is likely not stationary (ADF test p-value = {p_value}).\
    These anomaly detection algorithms will not perform well on non-stationary data.\
    Attempt to difference the time-series, compute the returns or use other methods\
    to make the time-series stationary before running this script.")
    elif p_value > 0.05: # attempt to make time-series stationary

        # find min p-value by using all the stationarity methods
        min_p_value = p_value
        min_index = None
        min_data = None

        for i in range(len(funcs_stationarity)):
            data_transf = funcs_stationarity[i][0](data)
            p_value = adfuller(data_transf)[1]

            if p_value < min_p_value:
                min_p_value = p_value
                min_index = i
                min_data = data_transf

        # transformation performed
        if not min_index is None:
            data = min_data
            stationarity_func_used = min_index
            offset_data = funcs_stationarity[min_index][1]

            # print results given min p-value
            if min_p_value <= 0.05:
                print(f"Time-series is now likely stationary by applying '{funcs_stationarity[min_index][2]}' function (p-value = {min_p_value}).")
            else:
                print(f"WARNING: unable to automatically make the time-series stationary (p-value = {min_p_value}). Using the lowest ADF p-value associated method: '{funcs_stationarity[min_index][2]}' function...")
        else: # no transformation performed
            print(f"WARNING: unable to automatically make the time-series stationary (p-value = {min_p_value}). Using the original time-series...")

# anomaly detection
anomalies_stdev = detect_anomalies_stdev(data,threshold_stdev) + [offset_data]
anomalies_autoencoder = detect_anomalies_autoencoder(data,threshold_autoencoder) + [offset_data]
anomalies_if = detect_anomalies_isolation_forest(data.reshape(-1, 1),contamination=contamination_isolation_forest) + [offset_data]
anomalies_lof = detect_anomalies_lof(data.reshape(-1, 1),contamination=contamination_lof) + [offset_data]

print(f"Anomalies detected with standard deviation method = {anomalies_stdev}")
print(f"Anomalies detected with autoencoder = {anomalies_autoencoder}")
print(f"Anomalies detected with isolation forest = {anomalies_if}")
print(f"Anomalies detected with local outlier factor = {anomalies_lof}")

# intersection
set_stdev = set(anomalies_stdev)
set_autoencoder = set(anomalies_autoencoder)
set_if = set(anomalies_if)
set_lof = set(anomalies_lof)

# find values that appear in two or more sets
common_values = list(set.intersection(set_stdev, set_autoencoder, set_if, set_lof))

print(f"Interception points between all methods = {sorted(common_values)}")

# common elements
def find_common_elements(min_common, *arrays):
    # Combine all arrays into one
    combined = np.concatenate(arrays)

    # Count the occurrences of each element
    counter = Counter(combined)

    # Find elements that appear in at least min_common arrays
    common_elements = {element for element, count in counter.items() if count >= min_common}

    return common_elements

common_elements = list(find_common_elements(2, anomalies_autoencoder, anomalies_if, anomalies_lof, anomalies_stdev))

print(f"Common elements between 2 or more methods: {sorted(common_elements)}")