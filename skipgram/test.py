import numpy as np
import collections
import random
import time
import codecs


# training parameters
learning_rate = 0.1
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

# evaluation parameters
valid_size = 20
valid_window = 100

#从词典的前100个词中随机选取20个词来验证模型
eval_words = np.random.choice(valid_window, valid_size, replace=False)

# word2vec parameters
embedding_size = 100
max_vocabulary_size = 50000
min_occurrence = 10 # 词典中出现的最低次数
skip_window = 2
num_skips = 4 # 每个输入中心词在其上下文区间中选取num_skips个词来生成样本
num_sampled = 64 # number of negative examples

def make_vocabulary(data):
    """

    data是一个一维list，可以是单个字也可以切分后的词
    data是将句子切分后再拼接而成

    只选取了max_vocabulary_size个词构建词典，对于出现过于稀疏的词不予考虑，一律变为UNK

    data_id: 词典中每个词的id
    word2id: 词和id的一一映射
    id2word: id和词的一一映射
    word2count: 每个词的出现次数
    vocabulary_size: 词典体积
    """
    word2count = [('UNK', -1)]

    word2count.extend(collections.Counter("".join(data))).most_common(max_vocabulary_size-1)

    for i in range(len(word2count), -1, -1):
        if word2count[i][1] < min_occurrence:
            word2count.pop(i)
        else:
            break

    vocabulary_size = len(word2count)

    word2id = dict()
    for i, (word, _) in enumerate(word2count):
        word2id[word] = i

    # 将data中的词转换为其对应的id
    data_id = list()
    unk_count = 0
    for word in data:
        index = word2id.get(word, 0)
        if index == 0:
            unk_count += 1

        data_id.append(index)

    word2count[0] = ('UNK', unk_count)
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return data_id, word2id, id2word, word2count, vocabulary_size

data_index = 0
def next_batch(batch_size, num_skips, skip_window, data):
    # data是上一模块生成的data_id
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # num_skips是每个中心词生成的样本数所以要求batch_size是其倍数
    # 每个中心生成的样本的label是从2*num_windows上下文中随机抽取的，所以num_skips<=2*skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1
    buf = collections.deque(maxlen=span) # 一个双向队列
    if data_index + span > len(data):
        data_index = 0
    buf.extend(data[data_index:data_index+span])
    data_index += span
    for i in range(batch_size // num_skips): # 每个batch中中心词的个数
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_words in enumerate(words_to_use):
            batch[i * num_skips + j] = buf[skip_window]
            labels[i * num_skips + j, 0] = buf[context_words]

        if data_index == len(data):
            buf.extend(data[0:span])
            data_index = span
        else:
            buf.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) -span) % len(data)

    return batch, labels

def train():
    s = time.process_time()
    print('load the data.......')
    f = open('wiki_s.txt_cut.txt')
    data = f.read()

    print('finish the loading, cost: {}'.format(str(time.process_time() - s)))

    s = time.process_time()
    print('make the vocabulary.......')
    data_id, word2id, id2word, word2count, vocabulary_size = make_vocabulary(data)
    print('finish making, cost: {}'.format(str(time.process_time() - s)))

    print('save vocabulary......')
    print('size of vocabulary: {}'.format(str(vocabulary_size)))
    with codecs.open('./vocabulary_text', 'w', 'utf-8') as f_out:
        L = sorted(list(word2id.items()), key=lambda x: int(x[1]))
        for word, id in L:
            f_out.write("%s\t%s\n"%(word, str(id)))

    





















