# Time-series anomaly detection
Detects anomalous data points in a time-series by various methods. Auto-stationarizes time-series.

# Executing the application

To install and execute this application, follow these steps:  

1. Clone this repository to your local machine:
   ```shell
   git clone https://github.com/leo-cb/anomaly-detection.git
   ```
   
2. Install requirements (developed with Python 3.11.4):
   ```shell
   pip install -r requirements.txt
   ```

3. Run the application:
   ```shell
   python detect-anomaly.py --file sample_data.csv
   ```

The application will output:
1. Status on time-series stationarity and success/failure on auto-stationarizing it by one of 4 methods: differencing, log differences, 2-differencing and cubic root.
2. The anomalous data points (indices) found by each one of 4 methods: Autoencoder, isolation forest, local outlier factor and median + standard deviation.
3. The common anomalous data points between all of these methods
4. The common anomalous data points between 2 or more of these methods

## Sample output

```shell
python detect-anomaly.py --file sample_data.csv
```

```
Time-series is now likely stationary by applying 'Log-differencing' function (p-value = 0.027494803292245483).
Anomalies detected with stdev method = [98]
Anomalies detected with autoencoder method = [98]
Anomalies detected with isolation_forest method = [ 1  2  3  4  5  6  7  8  9 10 11 96 97 98]
Anomalies detected with lof method = [98]
Common anomalies detected by all methods: [98]
Common elements between 2 or more methods: [98]
```

## Arguments

These arguments allow for customization of the algorithms and stationarity checking/imputing:  

```python
parser.add_argument('--file', type=str, required=True, help='Input the file path to a one-column csv file with no header with time-series values.')
parser.add_argument('--check_stationarity', type=bool, default=True)
parser.add_argument('--auto_stationarity', type=bool, default=True)
parser.add_argument('--anomaly_algorithm', nargs='+', default=["all"], choices=['all', 'autoencoder', 'isolation_forest', 'lof', 'stdev'], 
                    help='Choose one or more anomaly detection algorithms. Options are: "all", "autoencoder", "isolation_forest", "lof", "stdev"')
parser.add_argument('--threshold_autoencoder', type=float, default=2)
parser.add_argument('--contamination_isolation_forest', type=str, default="auto")
parser.add_argument('--contamination_lof', type=str, default="auto")
parser.add_argument('--threshold_stdev', type=float, default=2)
```
