python ../../wikidata_for_authors.py run ~/datasets/wikidata/index_enwiki-20190420.db \
    ~/datasets/wikidata/index_dewiki-20190420.db \
    ~/datasets/wikidata/torchbiggraph/wikidata_translation_v1.tsv.gz \
    ~/notebooks/bert-text-classification/authors.pickle \
    ~/notebooks/bert-text-classification/author2embedding.pickle

# OPTIONAL: Projector format
python ../../wikidata_for_authors.py convert_for_projector \
    ~/notebooks/bert-text-classification/author2embedding.pickle
    extras/author2embedding.projector.tsv \
    extras/author2embedding.projector_meta.tsv

