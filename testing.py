from word_embedding import WordEmbedding


filename = "data/glove.840B.300d.txt"
vocabulary_size = 100000

word_embedding = WordEmbedding(filename, True, vocabulary_size)
woman_vec = word_embedding.word_to_vec("woman")
manager_vec = word_embedding.word_to_vec("manager")
man_vec = word_embedding.word_to_vec("man")

tops = word_embedding.nearest_words_for_word_embedding(manager_vec - man_vec + woman_vec, topn=10)

