import numpy as np
from sklearn.preprocessing import StandardScaler
from lib.anomaly import detect_anomalies_autoencoder, detect_anomalies_isolation_forest,\
    detect_anomalies_lof, detect_anomalies_one_class_svm, detect_anomalies_stdev
from statsmodels.tsa.stattools import adfuller

# parameters
check_stationary = True
anomaly_algorithm = 0 # 0 - all, 1 - , 2 - , ...
threshold_autoencoder = 2
contamination_isolation_forest = "auto"
contamination_lof = "auto"
threshold_stdev = 2

# generate synthetic time series data with anomalies
np.random.seed(123)
n = 1000 # time-series length
data = np.random.randn(n) + np.sin(np.linspace(0, 6 * np.pi, n)) + 0.2 * np.random.randn(n)
data[200:250] += 10  # add anomalies

# normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 1))
data = data.squeeze()

# check if time-series is stationary
if check_stationary:
    result_adf = adfuller(data)
    p_value = result_adf[1]

    if p_value > 0.05:
        print("WARNING: the provided time-series is likely not stationary (ADF test p-value = {p_value}).\
            These anomaly detection algorithms will not perform well on non-stationary data.\
            Attempt to difference the time-series, compute the returns or use other methods\
            to make the time-series stationary before running this script.")

# anomaly detection
anomalies_stdev = detect_anomalies_stdev(data,threshold_stdev)
anomalies_autoencoder = detect_anomalies_autoencoder(data,threshold_autoencoder)
anomalies_if = detect_anomalies_isolation_forest(data.reshape(-1, 1),contamination=contamination_isolation_forest)
anomalies_lof = detect_anomalies_lof(data.reshape(-1, 1),contamination=contamination_lof)

print(f"Anomalies detected with standard deviation method = {detect_anomalies_stdev(data,threshold_stdev)}")
print(f"Anomalies detected with autoencoder = {detect_anomalies_autoencoder(data,5)}")
print(f"Anomalies detected with isolation forest = {detect_anomalies_isolation_forest(data.reshape(-1, 1),contamination=contamination_isolation_forest)}")
print(f"Anomalies detected with local outlier factor = {detect_anomalies_lof(data.reshape(-1, 1),contamination=contamination_lof)}")

# intersection
set_stdev = set(anomalies_stdev)
set_autoencoder = set(anomalies_autoencoder)
set_if = set(anomalies_if)
set_lof = set(anomalies_lof)

# find values that appear in two or more sets
common_values = list(set.intersection(set_stdev, set_autoencoder, set_if, set_lof))

print(f"Interception points between all methods = {sorted(common_values)}")

from collections import Counter

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