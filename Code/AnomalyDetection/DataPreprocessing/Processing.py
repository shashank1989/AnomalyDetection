import pandas as pd
from LogParsing.Drain import *
from LogParsing.Session import *
import numpy as np
import gc

class Processing:
    
    def __init__(self,df_obj):
        self.df_obj = df_obj
    
    def process_data(self,df):
        df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
        df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0,inplace=True) # This will capture the temporal difference
        return df
    
    def file_generator(self,file, df, features):
        with open(file, 'w') as f:
            for _, row in df.iterrows():
                for val in zip(*row[features]):
                    f.write(','.join([str(v) for v in val]) + ' ')
                f.write('\n')
    
    def train_test_data(self,logdf,train_ratio,output_dir):
        
        #Normal Logs Training and Testing 
        
        print("\n Normal Logs Training and Testing\n")

        df_normal =logdf[logdf["Label"] == 0]
        df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
        normal_len = len(df_normal)
        train_len = int(normal_len * (1-train_ratio))

        # Train Normal Log
        train_normal = df_normal[:train_len]
        self.file_generator(os.path.join(output_dir,'train_normal'), train_normal, ["EventId"])
        print("Train normal log size {}".format(train_len))

        # Test Normal log
        test_normal = df_normal[train_len:]
        self.file_generator(os.path.join(output_dir, 'test_normal'), test_normal, ["EventId"])
        print("Test Normal size {}".format(normal_len - train_len))
        
        #Abnormal Logs Training and Testing 

        df_abnormal =logdf[logdf["Label"] == 1]
        df_abnormal = df_abnormal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
        abnormal_len = len(df_abnormal)
        train_len = int(abnormal_len * (1-train_ratio))

        # Train Abnormal Logs
        train_abnormal = df_abnormal[:train_len]
        self.file_generator(os.path.join(output_dir,'train_abnormal'), train_abnormal, ["EventId"])
        print("Train Abnormal logs size {}".format(train_len))

        # Test Abnormal logs
        test_abnormal = df_abnormal[train_len:]
        self.file_generator(os.path.join(output_dir, 'test_abnormal'), test_abnormal, ["EventId"])
        print("Test Abnormal logs size {}".format(abnormal_len - train_len))

        del df_abnormal
        del train_abnormal
        del test_abnormal
        gc.collect()
        
        print("\nValidate file path\n {}".format(os.path.join(output_dir,'train_abnormal')))
        
    def load_normal_data(self,output_dir):
        
        # Load train_normal and test_normal data
        
        train_test_dir = os.path.join(output_dir)
        print("Train & Test Directory {}".format(train_test_dir))

        train_normal = []

        with open(train_test_dir  + "/train_normal", "r") as f:
            for line in f:
                train_normal.append([ln.split(",")[0] for ln in line.split()])
        train_normal = np.array(train_normal).reshape(-1,1)

        test_normal = []
        
        with open(train_test_dir + "/test_normal", "r") as f:
            for line in f:
                test_normal.append([ln.split(",")[0] for ln in line.split()])
        test_normal = np.array(test_normal).reshape(-1,1)
        
        return train_normal,test_normal
    
    def load_abnormal_data(self,output_dir):
        
        # Load train_abnormal and test_abnormal data
        
        train_test_dir = os.path.join(output_dir)
        print("Train & Test Directory {}".format(train_test_dir))

        train_abnormal = []

        with open(train_test_dir  + "/train_abnormal", "r") as f:
            for line in f:
                train_abnormal.append([ln.split(",")[0] for ln in line.split()])
        train_abnormal = np.array(train_abnormal).reshape(-1,1)

        test_abnormal = []
        with open(train_test_dir + "/test_abnormal", "r") as f:
            for line in f:
                test_abnormal.append([ln.split(",")[0] for ln in line.split()])
        test_abnormal = np.array(test_abnormal).reshape(-1,1)
        
        return train_abnormal,test_abnormal
    
    def consolidate_normal_abnormal_logs(self,normal,abnormal,train_test_type):
         # combine normal and abnormal dataset
        print("{} Normal and Abnormal dataset".format(train_test_type))
        X = np.vstack((normal, abnormal))
        Y = np.vstack((np.zeros(normal.shape), np.ones(abnormal.shape)))
        X = X.squeeze()
        Y = Y.squeeze()
        return X,Y
    
    def convert_stoi(self,str_list, event_dict):
        return [event_dict.get(s, 0) for s in str_list]
    
    def convert_eventidTonumber(self,log_file,x_train_test_type,output_dir):
        
        log_event = pd.read_csv(output_dir + "/" +log_file + "_templates.csv") # read data from template file
        
        eventids = log_event["EventId"].tolist()
        
        print("Total logkey(exclude 0:UNK)", len(eventids)) # log Keys are the evenid's
        
        # 0 means unknown eventids
        
        event_dict = {eid: idx + 1 for idx, eid in enumerate(eventids)}
        
        #print("\nEvent_dict\n", event_dict)

        # convert into EventId into number
        for idx in range(x_train_test_type.shape[0]):
            x_train_test_type[idx] = self.convert_stoi(x_train_test_type[idx], event_dict)
        
        return x_train_test_type
        
        