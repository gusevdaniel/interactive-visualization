import numpy as np
import pandas as pd

from SPARQLWrapper import SPARQLWrapper, JSON

from utils import *


def get_classes():
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?subClass ?class
        WHERE  { ?subClass rdfs:subClassOf  ?class;
        rdfs:label ?label
        FILTER ((regex(?class, "http://dbpedia.org/ontology/") || regex(?class, "http://www.w3.org/2002/07/owl")) && lang(?label) = "en")
        } 
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    subClasses = dict()
    for result in results["results"]["bindings"]:
        subClass = delete_host(result["subClass"]["value"])
        ontClass = delete_host(result["class"]["value"])
        subClasses[subClass] = ontClass
    return subClasses


def subClassOf(subClasses, class_1, class_2):
    answer = False
    while True:
        if class_1 == class_2:
            answer = True
            break
        else:
            if class_1 not in subClasses:
                break
            class_1 = subClasses[class_1]
    return answer


def upper_classes(subClasses, some_class):
    classes = set()
    while True:
        classes.add(some_class)
        if some_class not in subClasses:
            break
        some_class = subClasses[some_class]
    return classes


def low_class(subClasses, classes):
    lc = None
    if len(classes) == 1:
        lc = classes.pop() 
    for elem in classes:
        answers = set()
        other_classes = set.copy(classes)
        other_classes.remove(elem)
        for other_elem in other_classes:
            answer = subClassOf(subClasses, elem, other_elem)
            answers.add(answer)

        if True in answers and len(answers) == 1:
            lc = elem
            break

    return lc


def joint_class(subClasses, class_1, class_2):
    classes_1 = upper_classes(subClasses, class_1)
    classes_2 = upper_classes(subClasses, class_2)

    classes = set.intersection(classes_1, classes_2)
    classes.discard('owl#Thing')
    lower_class = low_class(subClasses, classes)

    return lower_class


def determine_types(subClasses, df):
    en_df = df.loc[df['lang'] == 'en']
    indexes = list(en_df['ent1_id'])

    for ind in indexes:
        row1 = df.loc[df['ent1_id'] == ind]
        row2 = df.loc[df['ent2_id'] == ind]

        type_1 = row1['type'].values[0]
        type_2 = row2['type'].values[0]

        if type_1 != type_2:
            if type_1 == 'Book':
                type_1 = type_2
            elif type_2 == 'Book':
                type_2 = type_1
            elif subClassOf(subClasses, type_1, type_2):
                type_2 = type_1
            elif subClassOf(subClasses, type_2, type_1):
                type_1 = type_2
            else:
                ont = joint_class(subClasses, type_1, type_2)
                if ont != None:
                    type_1 = ont
                    type_2 = ont
                else:
                    type_2 = type_1

            id_1 = row1.index[0]
            id_2 = row2.index[0]

            df.loc[id_1,'type'] = type_1
            df.loc[id_2,'type'] = type_2

    return df


def calculate_alignment(df):
    counter = 0
    en_df = df.loc[df['lang'] == 'en']
    indexes = list(en_df['ent1_id'])

    for ind in indexes:
        row1 = df.loc[df['ent1_id'] == ind]
        row2 = df.loc[df['ent2_id'] == ind]
        type_1 = row1['type'].values[0]
        type_2 = row2['type'].values[0]

        if type_1 == type_2:
            counter += 1

    result = counter / len(indexes)
    result = int(result * 10000) / 100
    return result


def align_types(df):
    print(f'Before: {calculate_alignment(df)} %')
    subClasses = get_classes()
    df = determine_types(subClasses, df)
    result = calculate_alignment(df)
    print(f'After: {result} %')
    assert result == 100.0

    types = df['type'].unique()
    for type_ in types:
        selected_df = df.loc[df['type'] == type_]
        number_ents = len(selected_df)
        assert (number_ents % 2) == 0

    return df


if __name__ == '__main__':
    filename = 'MultiKE_Word2Vec_EN_RU'

    df = pd.read_csv(filename + '.csv')
    df = align_types(df)

    filename = filename + '_aligned.csv'
    df.to_csv(filename, index=False)