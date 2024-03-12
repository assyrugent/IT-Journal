from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, WordEmbeddings, CharacterEmbeddings, BytePairEmbeddings, BertEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from tqdm import tqdm


# IMPORTANT: COMMENT OUT THE TRAINING PROCESSES YOU DON'T NEED!
# If you don't do that, this will take a long time.

# Define where the training data folders are located
folder_location = os.path.dirname(__file__)
file_name = 'transliteration_PoS'
fold_prefix = 'fold0'
folds = [0,1,2,3,4]


# We list here the different embeddings we tested for the paper
embeddings = [
    StackedEmbeddings( # Embedding 1
        [
            WordEmbeddings('glove'),
            CharacterEmbeddings()
        ]
    ),
    BytePairEmbeddings( # Embedding 2
        'multi'
    ),
    StackedEmbeddings( # Embedding 3
        [
            CharacterEmbeddings(),
            FlairEmbeddings('multi-forward'),
            FlairEmbeddings('multi-backward')
        ]
    ),
    TransformerWordEmbeddings('bert-base-multilingual-cased') # Embedding 4
]

# Iterate through the pre-trained embeddings model
for embedding in embeddings:
    # Iterate through the folds of data
    for fold in tqdm(folds):
        # Define parameters necessary for the training    
        dictionary = []
        tag_dict = defaultdict(int)
        tag_type = 'pos'
        columns = {0: 'text', 1: 'pos'}
        # The data_folder is where the data of the current training is found
        data_folder = f'{folder_location}/{file_name}_{fold_prefix}{fold}'
        # We here define the training and testing files
        corpus: Corpus = ColumnCorpus(data_folder, columns,
                                      train_file='train.txt',
                                      test_file='test.txt')
        # Define the sequence tagger parameters
        tagger = SequenceTagger(hidden_size=256,
                                embeddings=embedding,
                                tag_dictionary=corpus.make_tag_dictionary('pos'),
                                tag_type='pos',
                                use_crf=True,
                                )
        # Initiate the sequence tagger parameters and the corpus
        trainer = ModelTrainer(tagger, corpus)
        # Train the sequence tagger
        trainer.train(f'{folder_location}/{file_name}_{fold_prefix}{fold}/{embeddings.index(embedding)}/model', learning_rate=0.1, 
            mini_batch_size=8, 
            anneal_factor=0.5, 
            patience=5, 
            max_epochs=100
            )