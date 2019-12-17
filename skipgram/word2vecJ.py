import math
import sys
import numpy as np

class Ngram:
    def __init__(self, tokens):
        self.tokens = tokens
        self.count = 0
        self.score = 0.0

    def set_score(self, score):
        self.score = score

    def get_string(self):
        return '_'.join(self.tokens)

class Corpus:
    def __init__(self, filename, word_phrase_passes, word_phrase_delta, word_phrase_threshold, word_phrase_filename):
        i = 0
        file_pointer = open(filename, 'r')

        all_tokens = []

        for line in file_pointer:
            line_tokens = line.split()
            for token in line_tokens:
                #token = token.lower()

                if len(token) > 1 and self.isChinese(token):
                    all_tokens.append(token)

                i += 1
                if i % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\rReading corpis: %d" % i)

        sys.stdout.flush()
        print("\rCorpus read: %d" % i)

        file_pointer.close()

        self.tokens = all_tokens # all words

        for x in range(1, word_phrase_passes + 1):
            self.build_grams(x, word_phrase_delta, word_phrase_threshold, word_phrase_filename)

        self.save_to_file(filename)

    def isChinese(self, s):
        for c in s:
            if not ('\u4e00' <= c <= '\u9fa5'):
                return False
        return True

    def build_grams(self, x, word_phrase_delta, word_phrase_threshold, word_phrase_filename):
        ngrams = [] # 存储每个ngram
        ngrams_map = {} # 存储每个ngram在ngrams数组中的位置

        # 计算每个词或词组的数量
        token_count_map = {}
        for token in self.tokens:
            if token not in token_count_map:
                token_count_map[token] = 1
            else:
                token_count_map[token] += 1

        # 搭建每个词和其后面那个词组成的词对
        i = 0
        ngram_l = [] # 滑动窗口
        for token in self.tokens:
            if len(ngram_l) == 2:
                ngram_l.pop(0)

            ngram_l.append(token)
            ngram_t = tuple(ngram_l)

            if ngram_t not in ngrams_map:
                ngrams_map[ngram_t] = len(ngrams)
                ngrams.append(Ngram(ngram_t))
            ngrams[ngrams_map[ngram_t]].count += 1

            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rBuilding n-grams (%d pass): %d" % (x, i))

        sys.stdout.flush()
        print("\rn-grams (%d pass) built: %d" % (x, i))

        # 计算score并且初次过滤,过滤出关系比较大的词组
        filtered_ngrams_map = {}
        file_pointer = open(word_phrase_filename + ('-%d' % x), 'w')

        i = 0
        for ngram in ngrams:
            product = 1
            for word_string in ngram.tokens:
                product *= token_count_map[word_string]
            ngram.set_score((float(ngram.count) - word_phrase_delta) / float(product))

            if ngram.score > word_phrase_threshold:
                filtered_ngrams_map[ngram.get_string()] = ngram
                file_pointer.write('%s %d\n' % (ngram.get_string(), ngram.count))

            i += 1
            if i % 10000:
                sys.stdout.flush()
                sys.stdout.write('\rScoring n-grams: %d' % i)

        sys.stdout.flush()
        print('\rScored n-grams: %d, filtered n-grams: %d' % (i, len(filtered_ngrams_map)))
        file_pointer.close()

        all_tokens = []
        i = 0

        while i < len(self.tokens):
            if i + 1 < len(self.tokens):
                ngram_l = []
                ngram_l.append(self.tokens[i])
                ngram_l.append(self.tokens[i+1])
                ngram_string = '_'.join(ngram_l)

                if len(ngram_l) == 2 and (ngram_string in filtered_ngrams_map): # 如果相邻的两个token存活下来了则加进词库中
                    ngram = filtered_ngrams_map[ngram_string]
                    all_tokens.append(ngram.get_string())
                    i += 2
                else:
                    all_tokens.append(self.tokens[i])
                    i += 1

            else:
                all_tokens.append(self.tokens[i])
                i += 1

        print('tokens combined')

        self.tokens = all_tokens

    def save_to_file(self, filename):
        i = 1
        filepointer = open('preprocessed-' + filename, 'w')
        line = ''
        for token in self.tokens:
            if i % 20 == 0:
                line += token
                filepointer.write('%s\n' % line)
                line = ''
            else:
                line += token + ' '

            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rWriting to preprocessed input file')

        sys.stdout.flush()
        print('\rPreprocess input file written')

        filepointer.close()

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0

class Vocabulary:
    def __init__(self, corpus, min_count):
        self.words = []
        self.word_map = {}
        self.build_words(corpus)

        self.filter_for_rare_and_common(min_count)

    def build_words(self, corpus):
        words = []
        word_map = {}

        i = 0
        for token in corpus:
            if token not in word_map:
                word_map[token] = len(words) # 每个token在word中的位置
                words.append(Word(token))
            words[word_map[token]].count += 1

            i += 1
            if i % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rBuilding vocabulary: %d' % len(words))

        sys.stdout.flush()
        print('\rVocabulary built: %d' % len(words))

        self.words = words
        self.word_map = word_map

    def filter_for_rare_and_common(self, min_count):
        # remove rare words and sort
        tmp = []
        tmp.append(Word('{rare}'))
        unk_hash = 0

        count_unk = 0
        for token in self.words:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        word_map = {}
        for i, token in enumerate(tmp):
            word_map[token.word] = i

        self.words = tmp
        self.word_map = word_map
        pass

    def indices(self, tokens):
        return [self.word_map[token] if token in self else self.word_map['{rare}'] for token in tokens]

    def __getitem__(self, item):
        return self.words[item]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __contains__(self, item):
        return item in self.word_map

class TableForNegativeSamples:
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # normalizing costants

        table_size = 100000000
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1/ (1 + math.exp(-z))


def save(vocab, nn0, filename):
    file_pointer = open(filename, 'w')
    for token, vector in zip(vocab, nn0):
        word = token.word.replace(' ', '_')
        vector_str = ' '.join([str(s) for s in vector])
        file_pointer.write('%s %s\n' % (word, vector_str))

    file_pointer.close()

def load(filename):
    file_pointer = open(filename)
    words = {}
    for line in file_pointer:
        line = line.split()
        word = line[0]
        vec = []
        for i in range(1, len(line)):
            vec.append(line[i])
        words[word] = vec
        if len(words) % 10000:
            sys.stdout.flush()
            sys.stdout.write('Reading word vectors: %d' % len(words))

    sys.stdout.flush()
    print('model read successed!')

    return words

def similarity(str1, str2, filename):
    file_pointer = open(filename)
    vectors = []
    for line in file_pointer:
        if len(vectors) == 2:
            break
        line = line.split()
        word = line[0]
        if word == str1 or word == str2:
            vec = []
            for i in range(1, len(line)):
                vec.append(float(line[i]))
            vectors.append(vec)
        else:
            continue
    #print(vectors)
    if len(vectors) < 2:
        return None
    else:
        vec1 = np.array(vectors[0])
        vec2 = np.array(vectors[1])
        return np.sqrt(np.sum((vec1-vec2)**2))


if __name__ == '__main__':

    for input_filename in ['wiki_s_small.txt_cut.txt']:

        k_negative_sampling = 5
        min_count = 3
        word_phrase_passes = 3
        word_phrase_delta = 3
        word_phrase_threshold = 1e-4

        corpus = Corpus(input_filename, word_phrase_passes, word_phrase_delta, word_phrase_threshold, 'phrases-%s' % input_filename)
        vocab = Vocabulary(corpus, min_count) # 二次采样
        table = TableForNegativeSamples(vocab)

        for window in [2]:
            for dim in [100]:

                print('Training: %s-%d-%d-%d' % (input_filename, window, dim, word_phrase_passes))

                nn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(len(vocab), dim))
                nn1 = np.zeros(shape=(len(vocab), dim))

                initial_alpha = 0.01

                global_word_count = 0
                alpha = initial_alpha
                word_count = 0
                last_word_count = 0

                tokens = vocab.indices(corpus)

                for token_idx, token in enumerate(tokens):
                    if word_count % 10000 == 0:
                        global_word_count += (word_count - last_word_count)
                        last_word_count = word_count

                        sys.stdout.flush()
                        sys.stdout.write('\rTraining: %d of %d' % (global_word_count, len(corpus)))

                    current_window = np.random.randint(low=1, high=window+1) # 小技巧,使其更关注接近中心词的词
                    context_start = max(token_idx - current_window, 0) # 取前面的单词
                    context_end = min(token_idx + current_window + 1, len(tokens)) # 取后面的单词
                    context = tokens[context_start:token_idx] + tokens[token_idx+1:context_end]

                    print(context)
                    for context_word in context:
                        neu1e = np.zeros(dim) # 1*100 [0]
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(k_negative_sampling)]
                        for target, label in classifiers:
                            z = np.dot(nn0[context_word], nn1[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g*nn1[target]
                            nn1[target] += g * nn0[context_word]

                        nn0[context_word] += neu1e

                    word_count += 1

                global_word_count += (word_count - last_word_count)
                sys.stdout.flush()
                print('\rTraining finished: %d' % global_word_count)

                save(vocab, nn0, 'output-%s-%d-%d-%d' % (input_filename, window, dim, word_phrase_passes))























