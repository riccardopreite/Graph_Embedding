import os, pickle
directory = 'dumped'

def load_pickle():
    film2embedding = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f, 'rb') as fp:
            film2embedding.update(pickle.load(fp))
    print(film2embedding)
    with open("film2embedding.pickle",'wb') as out:
        pickle.dump(film2embedding, out)

load_pickle()
