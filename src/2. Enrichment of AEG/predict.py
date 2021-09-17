import os
import json

pred_file = '../data/eventCoref'
out_file = '../out/eventCoref'

pred_lines = [l.strip() for l in open(pred_file + '/pred.csv', encoding='UTF-8')]
chunk_size = 10000000
chunks = len(pred_lines) // chunk_size + (1 if len(pred_lines) % chunk_size else 0)

for i in range(chunks):
    print('write to part{}'.format(i))
    path = pred_file + '-part{}'.format(i)
    if not os.path.exists(path):
        os.mkdir(path)
    os.system('cp {}/events.json {}/'.format(pred_file, path))
    open(path + '/pred.csv', 'w', encoding='UTF-8').write('\n'.join(pred_lines[i * chunk_size:(i + 1) * chunk_size]))
    os.system('rm -f {}/cache* >> /dev/null'.format(path))
    out_path = '../out/eventCoref-part{}'.format(i)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for file in ['txt', 'json', 'bin']:
        os.system('cp {}/*.{} {}/'.format(out_file, file, out_path))
