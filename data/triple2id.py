import sys
from tqdm import tqdm

dataset = sys.argv[1]

entity2id = {}
with open('{}/entities.dict'.format(dataset)) as fd:
    for line in fd:
        if not line: continue
        idx, entity = line.strip().split('\t')
        entity2id[entity] = int(idx)
relation2id = {}
with open('{}/relations.dict'.format(dataset)) as fd:
    for line in fd:
        if not line: continue
        idx, relation = line.strip().split('\t')
        relation2id[relation] = int(idx)

for prefix in ['train', 'valid', 'test']:
    fw = open('{}/{}_id.txt'.format(dataset, prefix), mode='w')
    with open('{}/{}.txt'.format(dataset, prefix)) as fd:
        for line in tqdm(fd):
            if not line: continue
            items = line.strip().split('\t')
            h, r, t = items
            h = entity2id[h]
            r = relation2id[r]
            t = entity2id[t]
            fw.write('{}\t{}\t{}\n'.format(h, r, t))
    fw.close()
