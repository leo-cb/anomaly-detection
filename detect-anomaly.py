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
parser.add_argument('--check_stationarity', type=bool, default=True, help="If set to True, program checks for a trend in the time-series.")
parser.add_argument('--auto_stationarity', type=bool, default=True, help="If set to True, program attempts to de-trend time-series if a trend is detected.")
parser.add_argument('--anomaly_algorithm', nargs='+', default=["all"], choices=['all', 'autoencoder', 'isolation_forest', 'lof', 'stdev'], 
                    help='Choose one or more anomaly detection algorithms. Options are: "all", "autoencoder", "isolation_forest", "lof", "stdev"')
parser.add_argument('--threshold_autoencoder', type=float, default=2, help="Multiplier of data's standard deviation above the mean loss to consider data point as an anomaly.")
parser.add_argument('--contamination_isolation_forest', type=str, default="auto", help='Contamination parameter for isolation forest. Options are: "auto", or a number between ]0,0.5].')
parser.add_argument('--contamination_lof', type=str, default="auto", help='Contamination parameter for LOF. Options are: "auto", or a number between ]0,0.5].')
parser.add_argument('--threshold_stdev', type=float, default=2, help="Multiplier of data's standard deviation above the median to consider data point as an anomaly.")

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
if ('isolation_forest' in anomaly_algorithm or 'all' in anomaly_algorithm) and contamination_isolation_forest != "auto":
    try:
        contamination_isolation_forest = float(contamination_isolation_forest)
        if contamination_isolation_forest <= 0 or contamination_isolation_forest > 0.5:
            raise ValueError(f"contamination_isolation_forest must be in the range (0,0.5]: {contamination_isolation_forest}")
    except Exception as e:
        raise ValueError(f"contamination_isolation_forest must either be 'auto' or a float in the range (0,0.5]: {str(e)}")
    
if ('lof' in anomaly_algorithm or 'all' in anomaly_algorithm) in anomaly_algorithm and contamination_lof != "auto":
    try:
        contamination_lof = float(contamination_lof)
        if contamination_lof <= 0 or contamination_lof > 0.5:
            raise ValueError(f"contamination_lof must be in the range (0,0.5]: {contamination_lof}")
    except Exception as e:
        raise ValueError(f"contamination_lof must either be 'auto' or a float in the range (0,0.5]: {str(e)}")

# validate file
if not file.endswith(".csv"):
    raise ValueError("The input file must be a csv file.")

if not os.path.isfile(file):
    raise FileNotFoundError("Unable to find the input file.")

# load file
df = pd.read_csv(file,header=None)

if len(df.columns) != 1:
    raise Exception("The input file must have exactly one column.")

# convert to numeric and ignore non-numeric values
len_before = len(df)
df = pd.to_numeric(df.squeeze(), errors='coerce')
len_after = len(df)

if len_before > len_after:
    print(f"Removed {len_before - len_after} rows with non-numeric values.")

# check for NA values
if df.isnull().values.any():
    print("Non-numeric or NA values found and ignored.")

# drop NA values
len_before = len(df)
df = df.dropna()
len_after = len(df)
if len_before > len_after:
    print(f"Removed {len_before - len_after} rows with NA values.")


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

# define the anomaly detection functions with their respective arguments
anomaly_detectors = {
    'stdev': (detect_anomalies_stdev, [data, threshold_stdev]),
    'autoencoder': (detect_anomalies_autoencoder, [data, threshold_autoencoder]),
    'isolation_forest': (detect_anomalies_isolation_forest, [data.reshape(-1, 1), contamination_isolation_forest]),
    'lof': (detect_anomalies_lof, [data.reshape(-1, 1), contamination_lof])
}

# initialize a dictionary to store the anomalies detected by each method
anomalies = {}

print(f"offset data = {offset_data}")
offset_data = 0

# loop over the selected anomaly detection methods
for method in anomaly_algorithm:
    # if 'all' is selected, apply all methods
    if method == 'all':
        for name, (func, args) in anomaly_detectors.items():
            anomalies[name] = func(*args) + [offset_data]
        
        break
    # otherwise, apply only the selected methods
    elif method in anomaly_detectors:
        func, args = anomaly_detectors[method]
        anomalies[method] = func(*args) + [offset_data]

# print the anomalies detected by each method
for name, anomaly in anomalies.items():
    print(f"Anomalies detected with {name} method = {anomaly}")

# find the common anomalies detected by all methods
common_anomalies = set.intersection(*[set(anom) for anom in anomalies.values()])
print(f"Common anomalies detected by all methods: {sorted(common_anomalies)}")

# function to find common elements between 2 or more methods
def find_common_elements(min_common, *arrays):
    # Combine all arrays into one
    combined = np.concatenate(arrays)

    # Count the occurrences of each element
    counter = Counter(combined)

    # Find elements that appear in at least min_common arrays
    common_elements = {element for element, count in counter.items() if count >= min_common}

    return common_elements

# find common elements between 2 or more methods
common_elements = list(find_common_elements(2, *anomalies.values()))
print(f"Common elements between 2 or more methods: {sorted(common_elements)}")