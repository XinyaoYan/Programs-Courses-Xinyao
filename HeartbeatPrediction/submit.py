import pandas as pd

def max1min0(infilename="./data/submit.csv", outfilename="./data/submit_max1min0.csv"):
    data = pd.read_csv(infilename)
    print(data)
    for index, row in data.iterrows():
        row_max = max(list(row)[1:])
        if row_max > 0.9:
            for i in range(1, 5):
                if row[i] > 0.9:
                    data.iloc[index, i] = 1
                else:
                    data.iloc[index, i] = 0
    print(data)
    data.to_csv(outfilename, index=False)

def maxmax(infilename="./data/submit.csv", outfilename="./data/submit_maxmax.csv"):
    data = pd.read_csv(infilename)
    print(data)
    for index, row in data.iterrows():
        tmp = list(row)[1:]
        row_max_idx = tmp.index(max(tmp)) + 1
        for i in range(1, 5):
            if i == row_max_idx:
                data.iloc[index, i] = 1
            else:
                data.iloc[index, i] = 0
    print(data)
    data.to_csv(outfilename, index=False)

if __name__ == '__main__':
    # max1min0(infilename="lightgbm_tsfresh_submit.csv", outfilename="lightgbm_tsfresh_submit_max09min0.csv")
    # maxmax(infilename="lightgbm_tsfresh_submit.csv", outfilename="lightgbm_tsfresh_submit_maxmax.csv")
    max1min0(infilename="./data/submit.csv", outfilename="./data/submit_max1min0.csv")
    maxmax(infilename="./data/submit.csv", outfilename="./data/submit_maxmax.csv")