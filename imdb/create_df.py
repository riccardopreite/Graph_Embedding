
import pandas as pd
import csv


col = ['tt_id', 'text', 'is_positive', 'is_negative']
def create_df():
        '''Open train_merged.txt 
        0: positive/negative
        1: text
        2: tt_id
        '''
        tsv_train = open("train.tsv")
        tsv_dev = open("dev.tsv")
        train_tsv = csv.reader(tsv_train, delimiter="\t")
        dev_tsv = csv.reader(tsv_dev, delimiter="\t")
        new_train = [ [review[2], review[1], 1 if int(review[0])==1 else 0, 1 if int(review[0])==0 else 0] for review in train_tsv]
        new_dev = [ [review[2], review[1], 1 if int(review[0])==1 else 0, 1 if int(review[0])==0 else 0] for review in dev_tsv]
        train_df = pd.DataFrame(new_train, columns=col)
        dev_df = pd.DataFrame(new_dev, columns=col)
        train_df.to_pickle("imdb_train_df_meta.pickle")
        dev_df.to_pickle("imdb_dev_df_meta.pickle")
        return train_df, dev_df
create_df()
