import os, pickle
import numpy as np
import pandas as pd
from config import default_extra_cols
extras_dir = os.environ["EXTRAS_DIR"]
train_path = os.environ["TRAIN_DF_PATH"]

extra_cols=default_extra_cols
with_vec = True
with_gender = True
on_off_switch=False

def load_extras():
    gender_df = pd.read_csv(os.path.join(extras_dir, 'name_gender.csv'))
    author2gender = {
        row['name']: np.array([row['probability'], 0] if row['gender'] == 'M' else [0, row['probability']]) for
        idx, row in gender_df.iterrows()}

    with open(os.path.join(extras_dir, 'author2embedding.pickle'), 'rb') as f:
        author2vec = pickle.load(f)
    return author2gender, author2vec

def load():
    train = open(train_path, 'rb')
    df, _, _, _ = pickle.load(train)
    author2gender, author2vec = load_extras()

    if with_vec:
        AUTHOR_DIM = len(next(iter(author2vec.values())))

        if on_off_switch:
            AUTHOR_DIM += 1  # One additional dimension of binary (1/0) if embedding is available
    else:
        AUTHOR_DIM = 0
        
    if with_gender:
        GENDER_DIM = len(next(iter(author2gender.values())))
    else:
        GENDER_DIM = 0
        
    extras = np.zeros((len(df), len(extra_cols) + AUTHOR_DIM + GENDER_DIM))
    vec_found_selector = [False] * len(df)
    gender_found_selector = [False] * len(df)

    vec_found_count = 0 
    gender_found_count = 0



    for i, authors in enumerate(df['authors']):
            # simple extras
            extras[i][:len(extra_cols)] = df[extra_cols].values[i]

            # author vec
            if with_vec:
                for author in authors.split(';'):
                    if author in author2vec:
                        if on_off_switch:
                            extras[i][len(extra_cols):len(extra_cols) + AUTHOR_DIM - 1] = author2vec[author]
                            extras[i][len(extra_cols) + AUTHOR_DIM] = 1
                        else:
                            extras[i][len(extra_cols):len(extra_cols)+AUTHOR_DIM] = author2vec[author]

                        vec_found_count += 1
                        vec_found_selector[i] = True
                        break
            
            # author gender
            if with_gender:
                for author in authors.split(';'):
                    first_name = author.split(' ')[0]
                    if first_name in author2gender:
                        extras[i][len(extra_cols)+AUTHOR_DIM:] = author2gender[first_name]
                        gender_found_count += 1
                        gender_found_selector[i] = True
                        break

    return extras, vec_found_count, gender_found_count, vec_found_selector, gender_found_selector
