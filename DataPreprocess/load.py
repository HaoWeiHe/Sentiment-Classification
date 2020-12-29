import pandas as pd

class DataPreprocess():
    def load_data():
        data =[]
        f = open("YOUR_DATA_PATH","r")
        #here, we only use top 200000 data from review
        for _ in range(200000):
            line = f.readline()
            data.append(line)
            data = map(lambda x: x.rstrip(), data)
            data_json_str = "[" + ','.join(data) + "]"
            data_df = pd.read_json(data_json_str)
            data_df.head(200000).to_csv("YOUR_OUTPUT_PATH")

if __name__ == '__main__':
    dp = DataPreprocess()
    dp.load_data()
