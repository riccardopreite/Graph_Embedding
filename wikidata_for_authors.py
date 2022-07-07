"""

Prepare the Wikidata knowledge graph embeddings

# Extract embeddings for authors
python wikidata_for_authors.py run ~/datasets/wikidata/index_enwiki-20190420.db \
    ~/datasets/wikidata/index_dewiki-20190420.db \
    ~/datasets/wikidata/torchbiggraph/wikidata_translation_v1.tsv.gz \
    ~/notebooks/bert-text-classification/authors.pickle \
    ~/notebooks/bert-text-classification/author2embedding.pickle

Found 3684 QIDs for authors (not found: 11779)

# Convert for projector
python wikidata_for_authors.py convert_for_projector \
    ~/notebooks/bert-text-classification/author2embedding.pickle
    extras/author2embedding.projector.tsv \
    extras/author2embedding.projector_meta.tsv

"""
import pickle

import fire
import numpy as np
import pandas as pd
from wikimapper import WikiMapper
from smart_open import open
import multiprocessing as mp
from tqdm import tqdm
cols = []
review = {}
CORE_NUMBER = mp.cpu_count()-1
selected_entity_ids = set()
i = 0


def get_id():
    global i
    i+=1
    return str(i)
def run_sub_process(data, sub_process, col):
        step_size = len(data) // CORE_NUMBER
        pool = mp.Pool(mp.cpu_count())
        global i
        i = 0

        sub_data = [ {"id":get_id(), "data":data[index:index+step_size], "col":col} for index in range(0, len(data), step_size) ]
        print("Starting sub process:",sub_process)
        for result in tqdm(pool.imap(func=sub_process, iterable=sub_data), total=len(data)):
            pass
        
        
        # pool.starmap(sub_process, sub_data)
        pool.close()
        pool.join()

def dump_df(dictionary):
    pid = dictionary["id"]
    data = dictionary["data"]
    col = dictionary["col"]
    tmp = {}
    for index, line in data.iterrows():
        entity_id = line["tt_id"]
        new_row = np.array(line[col]).astype(float)

        if entity_id in selected_entity_ids:
            review_id = review[entity_id]
            if review_id not in tmp.keys():
                tmp[review[entity_id]] = new_row
            else:
                tmp[review[entity_id]] += new_row

    with open("dumped/data_"+pid+".pickle", 'wb') as f:
        pickle.dump(tmp, f)
        print("saved to dumped/data_"+pid+".pickle of len", str(len(tmp)))


def run_review(relation_train_path, review_train_path, out_file):


 # review_train_path = os.environ["REVIEW_PATH"]
    with open(review_train_path) as rw:
        for i, line in enumerate(rw):
            cols = line.split('\t')
            tt_id = cols[2].replace("\n","")
            review["film_"+tt_id] = tt_id
            selected_entity_ids.add("film_"+tt_id)

    film2embedding = {}
     # relation_train_path = os.environ["RELATION_PATH"]
    with open(relation_train_path) as re:
        df = pd.read_csv(re, sep='\t') 
        series = df.pop(df.columns[0])
        encoded = pd.get_dummies(df, prefix="embed")
        encoded["tt_id"] = series
        cols = list(encoded.columns)
        cols.remove("tt_id")
        run_sub_process(encoded, dump_df, cols)

def run_reviewold(relation_train_path, review_train_path, out_file):
# review_train_path = os.environ["REVIEW_PATH"]
    review = {}     
    selected_entity_ids = set() 

    with open(review_train_path) as rw:
        for i, line in enumerate(rw):
            cols = line.split('\t')
            tt_id = cols[2].replace("\n","")
            review["film_"+tt_id] = tt_id
            selected_entity_ids.add("film_"+tt_id)

    film2embedding = {}
    # relation_train_path = os.environ["RELATION_PATH"]
    with open(relation_train_path) as re:
        df = pd.read_csv(re, sep='\t') #,columns=["tt_id", "relation", "value"])

        series = df.pop(df.columns[0])
        encoded = pd.get_dummies(df, prefix="embed")
        #print(type(tt_id),type(dummies))
        #new_col = ["tt_id"].append(dummies.columns)
        #encoded = pd.DataFrame([tt_id,dummies], columns=new_col)
        encoded["tt_id"] = series
        cols = list(encoded.columns)
        cols.remove("tt_id")
        #for i, line in enumerate(encoded):
        for index, line in encoded.iterrows():
            #cols = line.split('\t')
            entity_id = line["tt_id"]#film_ttid
            if entity_id in selected_entity_ids:
                film2embedding[review[entity_id]] = np.array(encoded[cols]).astype(float)

            if not index % 100:
                print(f'Lines completed {index}')

    # Save
    with open(out_file, 'wb') as f:
        pickle.dump(film2embedding, f)

    print(f'Saved to {out_file}')

def run(wikimapper_index_en, wikimapper_index_de, graph_embedding_file, authors_file, out_file):
    """

    Find the correct Wikidata embeddings for authors in `author_file` and write them
    into a author2embedding mapping file.

    :param wikimapper_index_en:
    :param wikimapper_index_de:
    :param graph_embedding_file:
    :param authors_file:
    :param out_file:
    :return:
    """
    print('Starting...')

    with open(authors_file, 'rb') as f:
        authors_list = pickle.load(f)

    print('Author file loaded')


    en_mapper = WikiMapper(wikimapper_index_en)  # title language is defined in index file
    de_mapper = WikiMapper(wikimapper_index_de)  # title language is defined in index file

    print('WikiMapper loaded (de+en)')

    not_found = 0
    not_found_ = []

    selected_entity_ids = set()
    found = 0

    qid2author = {}

    for book_authors_str in authors_list:
        authors = book_authors_str.split(';')

        for author in authors:

            qid = None

            # Wikipedia article might have the occupation in parenthesis
            en_queries = [
                author,
                author.replace(' ', '_'),
                author.replace(' ', '_') + '_(novelist)',
                author.replace(' ', '_') + '_(poet)',
                author.replace(' ', '_') + '_(writer)',
                author.replace(' ', '_') + '_(author)',
                author.replace(' ', '_') + '_(journalist)',
                author.replace(' ', '_') + '_(artist)',
            ]
            for query in en_queries:  # Try all options
                qid = en_mapper.title_to_id(query)
                if qid is not None:
                    break

            # de
            if qid is None:
                de_queries = [
                    author,
                    author.replace(' ', '_'),
                    author.replace(' ', '_') + '_(Dichter)',
                    author.replace(' ', '_') + '_(Schriftsteller)',
                    author.replace(' ', '_') + '_(Autor)',
                    author.replace(' ', '_') + '_(Journalist)',
                    author.replace(' ', '_') + '_(Autorin)',
                ]
                for query in en_queries:  # Try all options
                    qid = de_mapper.title_to_id(query)
                    if qid is not None:
                        break

            if qid is None:
                not_found += 1
                not_found_.append(author)
            else:
                found += 1
                selected_entity_ids.add(qid)
                qid2author[qid] = author

    print(f'Found {len(selected_entity_ids)} QIDs for authors (not found: {not_found})')

    author2embedding = {}

    #with open(graph_embedding_file) as fp:
    with open(graph_embedding_file, encoding='utf-8') as fp:  # smart open can read .gz files
        for i, line in enumerate(fp):
            cols = line.split('\t')

            entity_id = cols[0]

            if entity_id.startswith('<http://www.wikidata.org/entity/Q') and entity_id.endswith('>'):
                entity_id = entity_id.replace('<http://www.wikidata.org/entity/', '').replace('>', '')

                if entity_id in selected_entity_ids:
                    author2embedding[qid2author[entity_id]] = np.array(cols[1:]).astype(np.float)

            if not i % 100000:
                print(cols[1:])
                print(f'Lines completed {i}')

    # Save
    with open(out_file, 'wb') as f:
        pickle.dump(author2embedding, f)

    print(f'Saved to {out_file}')


def convert_for_projector(author2embedding_path, out_projector_path, out_projector_meta_path):
    """

    Converts embeddings such that they can be visualized with Tensorflow Projector

    See http://projector.tensorflow.org/

    :param author2embedding_path: Path to output of `run()`
    :param out_projector_path: Write TSV file of vectors to this path.
    :param out_projector_meta_path: Write TSV file of metadata to this path.
    """
    print(f'Reading embeddings from {author2embedding_path}')

    with open(author2embedding_path, 'rb') as f:
        a2vec = pickle.load(f)

        vecs = []
        metas = []

        for a, vec in a2vec.items():
            vecs.append('\t'.join([str(t) for t in vec.tolist()]))
            metas.append(a)

        with open(out_projector_path, 'w') as ff:
            ff.write('\n'.join(vecs))

        with open(out_projector_meta_path, 'w') as ff:
            ff.write('\n'.join(metas))

    print(f'Saved tensors to {out_projector_path}')
    print(f'Saved meta info to {out_projector_meta_path}')


# Use full dump + filter for not found names
# https://github.com/maxlath/wikidata-filter

# Scrape https://www.randomhouse.de/Autoren/Uebersicht.rhd

if __name__ == '__main__':
    fire.Fire()
