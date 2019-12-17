import time
import jieba

def load_data(path):
    start = time.process_time()
    f = open(path)
    data = f.read()
    end = time.process_time()
    print('reading is over!, using of time:{}'.format(str(end-start)))

    data = data.replace("\n", "")
    return data

def cut_txt(path):
    start_read = time.process_time()
    print('reading......')
    f = open(path)
    text = f.read()
    end_read = time.process_time()
    print('reading is over!, using of time:{}'.format(str(end_read - start_read)))
    print('cutting the word......')
    start_cut = time.process_time()
    new_text = jieba.cut(text, cut_all=False)
    str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    end_cut = time.process_time()
    print('cutting is over!, using of time:{}'.format(str(end_cut - start_cut)))
    cut_file = path+'_cut.txt'
    f_out = open(cut_file, 'w', encoding='utf-8')
    f_out.write(str_out)
    f.close()
    f_out.close()
    print('writing is over!')

if __name__ == '__main__':
    cut_txt('./wiki_s_small.txt')
