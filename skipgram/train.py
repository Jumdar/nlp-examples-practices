from gensim.models import word2vec
import gensim
import logging

def train_model(file_name):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(file_name)
    model = gensim.models.Word2Vec(sentences, size=100, window=2, sg=1)
    model.save('./wiki_word2vec.model')
    model.wv.save_word2vec_format('./wiki_word2vec.bin', binary=True)


if __name__=='__main__':
    file_name = './wiki_s.txt_cut.txt'
    train_model(file_name)