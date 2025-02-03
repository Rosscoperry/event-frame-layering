import pandas as pd
import numpy as np
from tonic import Dataset, transforms
import torch
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def merge_event_csv(data_csv1, data_csv2, output_file):
    """
    Merge two eventbased csv files into one csv file. 
    
    Tested using Prophesee's eventbased data.
    
    """
    # load in data, specify column names 
    names = ["x", "y", "p", "t"]
    data_csv1 = pd.read_csv("East_5s.csv", header=None, names=names)
    data_csv2 = pd.read_csv("West_5s.csv", header=None, names=names)

    #convert dataframes to numpy arrays
    data_csv1.to_numpy(dtype=np.dtype([("x", int), ("y", int), ("p", int), ("t", int)]))
    data_csv2.to_numpy(dtype=np.dtype([("x", int), ("y", int), ("p", int), ("t", int)]))

    # add channel columns to track each layer 
    data_csv1['c'] = 0
    data_csv2['c'] = 1

    # compare lengths of the two channels - error if not equal
    if len(data_csv1) != len(data_csv2):
        logging.info("Channels are not the same length - c1: {}, c2: {}".format(len(data_csv1), len(data_csv2)))

        #cut arrays to the same length 
        min_len = min(len(data_csv1), len(data_csv2))
        data_csv1 = data_csv1[:min_len]
        data_csv2 = data_csv2[:min_len]
        
        #normalise arrays to t = 0 at position 0 
        data_csv1['t'] = data_csv1['t'] - data_csv1['t'].iloc[0]
        data_csv2['t'] = data_csv2['t'] - data_csv2['t'].iloc[0]
        
        logging.info("Channels cut to the same length: {} and the timestamps have been normalised".format(min_len))
        
    #merge the two channels 
    data_csv = pd.concat([data_csv1, data_csv2])
    data_csv.sort_values(by=['t'], inplace=True)

    data_csv.to_csv(output_file, index=False)
    
def visualise_eventbased_data(path_to_csv_data, num_channels = 2, accumulation_time_us=33333, frame_rate = 30):
    # load in data 
    csv_data = pd.read_csv(path_to_csv_data)
    
    # check number of channels, merged data should have a column name 'c'
    if 'c' not in csv_data.columns:
        err_msg = "No channel column found in csv, looking for a column name with 'c'"
        logging.error(err_msg)
        raise ValueError(err_msg)
    
    if num_channels != len(csv_data['c'].unique()):
        err_msg = "Number of channels in data does not match the number of channels specified expected {} channels, found {} channels".format(num_channels, len(csv_data['c'].unique()))
        logging.error(err_msg)
        raise ValueError(err_msg)
    
    # split data into time bins to use for frames
    
    # check timelength of data 
    time_length = csv_data['t'].iloc[-1] - csv_data['t'].iloc[0] 
    logging.info("Time length of data: {}".format(time_length))
    
    delta_t = 1000000 / frame_rate # time to generate a frame
    
    # create bin of time values with start and end times of each frame with delta_t
    time_bins = np.arange(csv_data['t'].iloc[0], csv_data['t'].iloc[-1], delta_t)
    
    print(time_bins)
    return 

    
    
    


merge_event_csv("East_5s.csv", "West_5s.csv", "merged.csv")

visualise_eventbased_data("merged.csv", 2, accumulation_time_us=1000000, frame_rate=30)


# function to visualize the data 

# function that begins pipeline towards training an snn


