import pandas as pd
from LogParsing.Drain import *
from LogParsing.Session import * 

class DataLoader:
    
    def __init__(self,filename):
        self.filename = filename
    
    def load_data(self):
        df = pd.read_csv(self.filename, sep='\t', header=None)
        df = df[0].str.split(' ', expand=True)
        print("File Size : ",len(df))
        print(df.head())
        return df
        
    def concatenate_with_none_values(self,row):
        return " ".join(filter(lambda x: x is not None, row))
    
    def combined_column(self,df):
        df["combined_Column"] = df.iloc[:, 9:].apply(self.concatenate_with_none_values, axis=1)
        data = df[[0,1,2,3,4,5,6,7,8,'combined_Column']]
        data.columns = ['label', 'id', 'date', 'code1', 'time', 'code2','component1','component2','level','content']
        print(data["content"].head(100))
        df_len = len(df)
        data_len = len(data)
        if df_len == data_len:
            print("Validation Check - Sizing After the Concatenation remain same , Before Concatenation :{} , After Concatenation {}".format(df_len,data_len))
        return data
    
    def skewness_analysis(self,data):
        print("label field value counts \n {}".format(data['label'].value_counts()))
        total_logs = len(data['label'])
        non_alerted_logs = (data['label'] == "-").sum()
        abnormal_logs = total_logs - non_alerted_logs
        print("\nTotal Data Size : {}, Alerted Anomaly Logs Size : {}, Non-Alerted Logs Size : {}".format(total_logs,abnormal_logs,non_alerted_logs))
        
    def cleaned_data(self,data):
        data.to_csv("BGL_Cleaned.log")
        
    def log_parser(self,input_dir, output_dir, log_file):
        #This function will use the Drain parser and generate the Structured and Event Template files
        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
        regex = [
            r'(0x)[0-9a-fA-F]+', #hexadecimal
            r'\d+.\d+.\d+.\d+',
            r'\d+'  # phone number in the format +CountryCode-AreaCode-PhoneNumber
        ]

        keep_para = False
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.4  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = LogParser(log_format,indir=input_dir,outdir=output_dir,depth=depth,st=st,rex=regex, keep_para=keep_para)
        parser.parse(log_file)
     

                
    def sliding_windows(self,df,window_size,step_size):
        return sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]],
                                para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60})
        
    
     