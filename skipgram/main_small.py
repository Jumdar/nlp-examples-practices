from word2vecJ import similarity

load_file = './wiki_word2vec.model'

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

model_file = open('output-wiki_s_small.txt_cut.txt-2-100-3')
line = model_file.readline()
model = 'output-wiki_s_small.txt_cut.txt-2-100-3'
f_out = open('./result.txt', 'w', encoding='utf-8')
for pair in test_data:

    res = similarity(pair[0], pair[1], model)
    if res == None:
        res = 'OOV'
    str_out = '' + pair[0] + ' ' + pair[1] + ' ' + str(res)
    print(str_out)
    f_out.write(str_out)
    f_out.write('\r\n')