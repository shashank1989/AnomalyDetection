from DataLoading.DataLoader import *
from DataPreprocessing.Processing import *
import os
import gc
from sklearn.utils import shuffle
import numpy as np

def get_train_test_data(current_directory):

    print("**************Data Loading and Parameter Setting*****************\n")

    data_dir = current_directory + "/data" # Currengt Data Directory
    output_dir = current_directory + "/output" # Output Directory
    log_file = "BGL.log" # Log File
    print(data_dir + "\n" + output_dir)
    dataobj = DataLoader(log_file) # Creating the object of DataLoader Class

    print("**************Return Processed Output Structured and template files after processing Drain Algorithm*****************\n")

    #dataobj.log_parser(data_dir, output_dir, log_file) # Creating structured and template file by using Drain Algorithm

    #Setting below functions for the sliding window implementation

    window_size = 5
    step_size = 1
    train_ratio = 0.4
    df = pd.read_csv(f'{output_dir}/{log_file}_structured.csv') # Reading Structured file which have addition column EventId and EventTemplate
    print("\n {}".format(df.head()))

    # Data Preprocessing
    print("\n********** Data Pre-Processing***************\n")
    processingobj = Processing(df)
    df = processingobj.process_data(df)
    print("Preprocessed Dataframe \n{}".format(df.head()))
    print("\nCheck Minimum ID Feature Value : \n {}".format(df[['Id']].min()))

    # Sliding Window Implementation

    print("\n Sliding Window implementation\n")
    logdf = dataobj.sliding_windows(df[["timestamp", "Label", "EventId", "deltaT"]],window_size,step_size)
    print("logdf head \n {}".format(logdf.head()))

    print("Window_size  = 300 , Step_size = 60")

    #Validate Window and stepsize
    print("Validate Windowsize , stepsize implementation min: {},max : {}, window_size : {}".format(np.min(logdf['timestamp'][0]),np.max(logdf['timestamp'][0]),np.max(logdf['timestamp'][0])-np.min(logdf['timestamp'][0])))

    #Segregating Normal and Abnormal data and storing the data to output directory

    print("\nSegregating Normal and Abnormal data and storing the data to output directory\n")

    processingobj.train_test_data(logdf,train_ratio,output_dir)

    #Loading Normal Data

    print("\nLoading Normal Data\n")

    train_normal,test_normal = processingobj.load_normal_data(output_dir)

    #Loading Abnormal Data

    print("\nLoading Abnormal Data\n")

    train_abnormal,test_abnormal = processingobj.load_abnormal_data(output_dir)

    #Consolidate Training normal and abnormal data

    print("\n Consolidate Normal and Abnormal logs - Training set : x_train and y_train\n")

    x_train,y_train = processingobj.consolidate_normal_abnormal_logs(train_normal,train_abnormal,"train")

    assert len(x_train) == len(y_train)

    print("x_train size : {},y_train size :{}".format(len(x_train),len(y_train)))

    #Convert EventidTonNumber Training data

    print("Convert EventidTonNumber Training data")

    x_train = processingobj.convert_eventidTonumber(log_file,x_train,output_dir)

    #Consolidate Test normal and abnormal data

    print("\n Consolidate Normal and Abnormal logs - Test set : x_test and y_test\n")

    x_test,y_test = processingobj.consolidate_normal_abnormal_logs(test_normal,test_abnormal,"test")

    assert len(x_test) == len(y_test)

    print("x_test size : {},y_test size :{}".format(len(x_test),len(y_test)))

    #Convert EventidTonNumber Test data

    print("Convert EventidTonNumber Test data")

    x_test = processingobj.convert_eventidTonumber(log_file,x_test,output_dir)

    print(x_test[1],y_test[1])
    print(x_train[1],y_train[1])

    rand_train_index = shuffle(np.arange(len(y_train)), random_state=88)
    
    x_train = x_train[rand_train_index]
    
    y_train = y_train[rand_train_index]

    rand_test_index = shuffle(np.arange(len(y_test)), random_state=88)
    
    x_test = x_test[rand_test_index]
  
    y_test = y_test[rand_test_index]

    (x_train, y_train), (x_test, y_test)

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    print("Length Validation assertion successful")

    return x_train,y_train,x_test,y_test

