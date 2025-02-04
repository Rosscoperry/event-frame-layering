import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


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

##### FINALISE COLOUR ARG ########
def visualise_eventbased_data(path_to_csv_data, num_channels = 2, accumulation_time_us=33333, frame_rate = 30, color=False):
    
    """
    Visualise eventbased data from a csv file.
    
    
    """
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
    
    # subtract accumulation times from each bin to get the start time of each frame
    time_bins_start = time_bins - accumulation_time_us
    time_bins_end = time_bins
    
    # create a list of frames with each frame containing the events that occur within the time bin into a numpy array
    frames = np.empty((len(time_bins_start), num_channels), dtype=object)
    for i in range(len(time_bins_start)):
        frame = csv_data[(csv_data['t'] >= time_bins_start[i]) & (csv_data['t'] < time_bins_end[i])]
        for channel in range(num_channels):
            frames[i, channel] = frame[frame['c'] == channel].to_numpy()
        
    # frames shape (number of frames, number of channels)
    logging.info("Frames shape: {}".format(frames.shape))
    
    # accumulate all the events in each frame with one colour for each channel
    res = (720, 1280, 3)

    # create a blank image append to video
    video = []
    
    # get colour list to represent value of pixel
    colours = [[255, 0, 0], [0, 255, 0]]

    for frame in frames:
        img = np.zeros(res, dtype=np.uint8)
        for channel in range(num_channels):
            for event in frame[channel]:
                rgb = colours[channel]
                img[event[1], event[0]] = rgb if event[2] == 1 else [0,0,0] # Check if event is on or off      
              
        video.append(img) 
        
      
    # combine list of numpy.ndarrays into a video saved to a file 
    logging.info("Creating video from frames")
    fig = plt.figure()
    ims = []
    for i in range(len(video)):
        im = plt.imshow(video[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=1000/frame_rate, blit=True, repeat_delay=1000) #interval in ms 
    ani.save('video.mp4')
    logging.info("Video saved as video.mp4")
    
    return video

def split_colours(number_of_colours):
    """
    creates an array of rgb colours to use in visualisation as specified by user
    """
    
    colours = []
    for i in range(number_of_colours):
        colours.append([np.random.randint(0, 255), np.random.randint(0, 255), 0])
    
    return colours


merge_event_csv("East_5s.csv", "West_5s.csv", "merged.csv")

visualise_eventbased_data("merged.csv", 2, accumulation_time_us=33333, frame_rate=30)


# function that begins pipeline towards training an snn


