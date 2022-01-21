import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from utils import *
from align_types import align_types


def dimension_decrease(X):
    tsne = TSNE(n_components=2, init='pca', random_state=123)
    X_decrease = tsne.fit_transform(X)
    return X_decrease


def form_pairs(source, results):
    links = dict()
    ent_links = source + 'ent_links'
    with open(ent_links, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            links[th[0]] = th[1]
                                  
    pairs = dict()
    kgs_ids = get_kgs_ids(results, inverse=True)
    for uri in links.keys():
        id1 = kgs_ids[uri]
        uri2 = links[uri]
        id2 = kgs_ids[uri2]
        pairs[id1] = id2
        pairs[id2] = id1
        
    return pairs


def determine_lang(ids):
    even_list = (ids % 2) == 0
    lang_list = []
    for elem in even_list:
        if elem == True:
            lang_list.append('en')
        else:
            lang_list.append('ln')
    return lang_list


def get_names(source, targets):
    names = list()
    for elem in targets:
        name = source[elem]
        name = delete_host(name)
        names.append(name)
    return names


def dict_types(filepath):
    types = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.replace('<', '')
            line = line.replace('>', '')
            line = line.strip().split(' ')
            if len(line) == 4:
                name = delete_host(line[0])
                type_ = delete_host(line[2])
                types[name] = type_
    return types


def double_dict(fpath_1, fpath_2):
    types_1 = dict_types(fpath_1)
    types_2 = dict_types(fpath_2)

    full_types = dict()
    full_types.update(types_1)
    full_types.update(types_2)

    return full_types


def prepare_data(source, results, ftp_1, ftp_2):
    embeds = np.load(results + 'ent_embeds.npy')

    print('Dimension decrease')
    embeds = dimension_decrease(embeds)
    prepared_data = pd.DataFrame(embeds, columns=['x', 'y'])
    
    print('Pairs definition')
    ids_1 = list(prepared_data.index)
    prepared_data['ent1_id'] = ids_1
    ref_pairs = form_pairs(source, results)
    prepared_data['ent2_id'] = [ref_pairs[x] for x in ids_1]

    print('Name definition')
    kgs_ids = get_kgs_ids(results)
    prepared_data['ent1'] = get_names(kgs_ids, prepared_data['ent1_id'])
    prepared_data['ent2'] = get_names(kgs_ids, prepared_data['ent2_id'])

    print('Language definition')
    prepared_data['lang'] = determine_lang(prepared_data['ent1_id'])

    print('Types definition')
    ent_types = double_dict(ftp_1, ftp_2)
    prepared_data['type'] = prepared_data['ent1'].map(ent_types)

    print('Types alignment')
    prepared_data = align_types(prepared_data)

    return prepared_data


if __name__ == '__main__':
    source_folderpath = 'C:\\my-data\\EN_RU_15K\\EN_RU_15K_V1\\'
    results_folderpath = 'C:\\my-data\\output\\multike\\20210809104150\\'  # MultiKE results #Word2Vec EN-RU
    in_types_1 = 'C:\\my-data\\instance_types\\instance_types_en.ttl'
    in_types_2 = 'C:\\my-data\\instance_types\\instance_types_ru.ttl'
    data_filename = 'MultiKE_Word2Vec_EN_RU'

    df = prepare_data(source_folderpath, results_folderpath, in_types_1, in_types_2)

    print('Saving csv')
    folder = '..\\data\\'
    path = folder + data_filename + '.csv'
    df.to_csv(path, index=False)