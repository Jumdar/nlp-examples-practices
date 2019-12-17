from gensim.models import word2vec


load_file = './wiki_word2vec.model'
model = word2vec.Word2Vec.load(load_file)

test_data = []
f = open('./pku_sim_test.txt')
line = f.readline()
line = line.strip('\n')
test_data.append(line.split('\t'))
while line:
    line = f.readline()
    line = line.strip('\n')
    test_data.append(line.split('\t'))

test_data.remove([''])
#print(test_data[499])


f_out = open('./result2.txt', 'w', encoding='utf-8')
for pair in test_data:
    if (pair[0] in model) and (pair[1] in model):
        res = model.similarity(pair[0], pair[1])
    else:
        res = 'OOV'
    str_out = '' + pair[0] + ' ' + pair[1] + ' ' + str(res)
    f_out.write(str_out)
    f_out.write('\r\n')
