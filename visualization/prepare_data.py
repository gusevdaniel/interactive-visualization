import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

def edit_string(string):
    string = string.strip('\n').split('/')

    return string[-1]

def loadIds(fn):
    pair = dict()
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            pair[int(th[1])]=edit_string(th[0])
    return pair

def get_kgs_ids(folderpath):
    kg_ids_1 = loadIds(folderpath + 'kg1_ent_ids')
    kg_ids_2 = loadIds(folderpath + 'kg2_ent_ids')

    kgs_ids = dict()
    kgs_ids.update(kg_ids_1)
    kgs_ids.update(kg_ids_2)

    return kgs_ids

def get_embeds(folderpath):
    ent_embeds = np.load(folderpath + 'ent_embeds.npy')
    df = pd.DataFrame(ent_embeds)

    return df

def dimension_decrease(df):
    X = df.values

    tsne = TSNE(n_components=2, init='pca', random_state=123)
    X_embedded = tsne.fit_transform(X)

    return X_embedded

def determine_lang(df):
    even_list = (df.index % 2) == 0

    lang_list = []
    for elem in even_list:
        if elem == True:
            lang_list.append('en')
        else:
            lang_list.append('ln')

    return lang_list

def form_pairs(pairspath):
    new_list = []
    with open(pairspath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            new_list.append(line)

    new_dict = {}
    for pair in new_list:
        ents = pair.split('\t')
        ent_0 = edit_string(ents[0])
        ent_1 = edit_string(ents[1])
        new_dict[ent_0] = ent_1
        new_dict[ent_1] = ent_0

    return new_dict

def dict_types(filepath):
    dict_types = {}

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.replace('<', '')
            line = line.replace('>', '')
            line = line.strip().split(' ')
            if len(line) == 4:
                ent_name    = edit_string(line[0])
                ent_type    = ent = edit_string(line[2])
                dict_types[ent_name] = ent_type

    return dict_types

def double_dict(filepath_0, filepath_1):
    dict_0 = dict_types(filepath_0)
    dict_1 = dict_types(filepath_1)

    full_types = dict()
    full_types.update(dict_0)
    full_types.update(dict_1)

    return full_types

def data_for_visualization(ent_links, embeds_folderpath, types_0_path, types_1_path):
    kgs_ids = get_kgs_ids(embeds_folderpath)
    df = get_embeds(embeds_folderpath)

    keys_list = [kgs_ids[x] for x in df.index]

    print('Dimension decrease')
    X_embedded = dimension_decrease(df)

    prepared_data = pd.DataFrame(X_embedded, columns=['x', 'y'])
    prepared_data['ent'] = keys_list

    print('Language definition')
    lang_list = determine_lang(prepared_data)
    prepared_data['lang'] = lang_list

    print('Pairs definition')
    pairs = form_pairs(ent_links)
    prepared_data['ent_tr'] = prepared_data['ent'].replace(pairs)

    print('Types definition')
    ent_types = double_dict(types_0_path, types_1_path)
    prepared_data['type'] = prepared_data['ent'].map(ent_types)

    return prepared_data

def embeds_to_csv(ent_links, embeds_folderpath, types_0_path, types_1_path, filename):
    vis_df = data_for_visualization(ent_links, embeds_folderpath, types_0_path, types_1_path)

    print('Saving csv')
    filename = filename + '.csv'
    vis_df.to_csv(filename, index=False)


if __name__ == "__main__":
    types_0 = 'C:\\my-data\\instance_types\\instance_types_en.ttl'
    types_1 = 'C:\\my-data\\instance_types\\instance_types_ru.ttl'
    ent_links = 'C:\\my-data\\EN_RU_15K\\EN_RU_15K_V1\\ent_links' # data for training

    # results_folder = 'C:\\my-data\\output\\multike\\20210809104150\\' # MultiKE results #Word2Vec EN-RU
    # csv_path = '..\\data\\MultiKE_Word2Vec_EN_RU'

    results_folder = 'C:\\my-data\\output\\multike\\20211011153818\\' # MultiKE results #Word2Vec EN-RU translated
    csv_path = '..\\data\\MultiKE_Word2Vec_EN_RU_translated'

    embeds_to_csv(ent_links, results_folder, types_0, types_1, csv_path)
