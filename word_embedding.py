from gensim import matutils
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import numpy as np


EMPTY = "<empty>"


# noinspection PyMethodMayBeStatic
class WordEmbedding:
    def __init__(self, filename, glove=False, vocabulary_size=10000, add_blank=False):
        """Create WordEmbedding from given file
        file behind filename: should be .txt or .word2vec for glove-models else .bin"""
        print("Start loading word-embedding-model: {}".format(filename))
        self.model = self._load_model(filename, glove, vocabulary_size)
        print("Finished loading.")

        if add_blank:
            self._add_blank_dimension()

    def _add_blank_dimension(self):
        # add one additional dimensions for the tag <empty>
        new_embedding_size = self.model.vector_size + 1
        new_syn = np.zeros((len(self.model.syn0), new_embedding_size), np.float32)
        for i, vector in enumerate(self.model.syn0):
            new_syn[i] = np.append(vector, 0)
        self.model.syn0 = new_syn

        self.blank_tag_embedding = np.zeros(new_embedding_size, np.float32)
        self.blank_tag_embedding[-1] = 1
        self.blank_tag_index = len(self.model.syn0)
        self.model.syn0 = np.concatenate((self.model.syn0, [self.blank_tag_embedding]))
        self.embedding_matrix = self.model.syn0
        self.model.vector_size = new_embedding_size
        # add to the vocabs
        self.model.vocab[EMPTY] = Vocab(index=self.blank_tag_index, count=None)
        self.model.index2word.append(EMPTY)

    def _load_model(self, filename, is_glove, vocabulary_size):
        if is_glove:
            filename = self._convert_glove(filename)
        return KeyedVectors.load_word2vec_format(filename, binary=(not is_glove), limit=vocabulary_size - 1,
                                                 datatype=np.float32)

    def _convert_glove(self, filename, force=False):
        file_ending = '.word2vec'
        if filename[-9:] == file_ending:  # file already in .word2vec format
            return filename
        output_name = filename + file_ending
        if not force:  # convert always if force
            directory = os.path.dirname(filename)
            if output_name in os.listdir(directory):  # convert only if output-file not yet found
                return output_name
        glove2word2vec(filename, output_name)  # convert to .word2vec
        return output_name

    def word_to_vec(self, word):
        try:
            return self.model.word_vec(word)
        except KeyError:
            return self._random_vector()

    def word_to_index(self, word):
        if word in self.model.vocab:
            return self.model.vocab[word].index
        else:
            return np.random.randint(0, len(self.model.vocab))

    def index_to_word(self, index):
        return self.model.index2word[index]

    def index_to_vec(self, index):
        return self.word_to_vec(self.index_to_word(index))

    def nearest_words_for_word_embedding(self, word_embedding, topn=1):
        nearest_words = self.model.similar_by_vector(word_embedding, topn=topn)
        print(nearest_words)
        most_similar_words = [word for word, _ in nearest_words]
        # noinspection PyUnboundLocalVariable
        print(most_similar_words)
        return most_similar_words[0] if topn == 1 else most_similar_words

    def _similarity_between_embeddings(self, word_embedding1, word_embedding2):
        return np.dot(matutils.unitvec(word_embedding1), matutils.unitvec(word_embedding2))

    def _random_vector(self):
        return matutils.unitvec(np.random.random((self.model.vector_size, )) - 0.5)
