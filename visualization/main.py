import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.models import HoverTool, WheelZoomTool
from bokeh.models import LabelSet, Select, Slider, MultiChoice
from bokeh.models.callbacks import CustomJS

from sklearn.neighbors import KDTree

COLUMNS = ['x', 'y', 'ent', 'lang', 'ent_tr', 'type', 'color', 'sizes', 'distance']

tooltips = [('Entity', '@ent' + ' (@lang)'), ('Type', '@type')]

def prepare_df():
    df = pd.read_csv(filepath)

    df['color'] = np.where(df['lang'] == 'en', "blue", "red")
    df['sizes'] = 5

    df['distance'] = np.nan

    return df

def df_to_cds(df):
    return ColumnDataSource(data=df)

def get_index(df, ent):
    ent_index = df.loc[df['ent'].isin([ent])].index[0]

    return ent_index

def set_color(df, ent_index):
    ent_lang = df.loc[ent_index]['lang']

    if ent_lang == 'en':
        df.at[ent_index, 'color'] = 'blue'
    else:
        df.at[ent_index, 'color'] = 'red'

    df.at[ent_index, 'sizes'] = 10

def df_update(df, ent):
    df['color'] = 'lightgray'
    df['sizes'] = 5

    if ent != 'None':
        ent_index = get_index(df, ent)
        set_color(df, ent_index)

        ent_tr = df.loc[ent_index]['ent_tr']
        try:
            ent_index = get_index(df, ent_tr)
            set_color(df, ent_index)
        except IndexError as error:
            pass
    else:
        df['color'] = np.where(df['lang'] == 'en', "blue", "red")

    return df_to_cds(df)

def find_neighbours(df, ent, ent_number):
    ent_id = get_index(df, ent)

    new_df = pd.DataFrame()
    new_df['x'] = df['x']
    new_df['y'] = df['y']
    ent_embeds = new_df.to_numpy()

    # save original indexes
    i = 0
    df_ids = {}
    np_ids = {}
    df_indexes = list(df.index)
    for elem in df_indexes:
        df_ids[elem] = i
        np_ids[i] = elem
        i += 1

    new_ent_id = df_ids[ent_id]

    kdt = KDTree(ent_embeds)
    kdt_distance, kdt_indexes = kdt.query([ent_embeds[new_ent_id]], k=ent_number)

    new_kdt_indexes = []
    for ind in kdt_indexes[0]:
        new_kdt_indexes.append(np_ids[ind])

    selected_df = df.loc[new_kdt_indexes]

    kdt_distance = kdt_distance.tolist()[0]
    selected_df['distance'] = kdt_distance

    return selected_df

def update_types():
    global df

    current_types = types_choice.value

    if 'All' in current_types or current_types == []:
        df = prepare_df()
        src = df_to_cds(df)
        source.data.update(src.data)
    else:
        df = prepare_df()
        df = df.loc[df['type'].isin(current_types)]
        src = df_to_cds(df)
        source.data.update(src.data)

    ents = list(df['ent'])
    ents.insert(0, 'None')

    ent_select.options = ents 
    ent_select.value = ents[0]
    source.selected.indices = []

def update():
    # entity
    ent = ent_select.value
    src = df_update(df, ent)
    source.data.update(src.data)

    # neighbours
    if ent != 'None':
        ent_number = ent_num_slider.value
        df2 = find_neighbours(df, ent, ent_number)

        src2 = ColumnDataSource(data=df2)
        source2.data.update(src2.data)
    else:
        df2 = pd.DataFrame(columns=COLUMNS)

        src2 = ColumnDataSource(data=df2)
        source2.data.update(src2.data)

# load data
folder = 'data\\'
# filename = 'MultiKE_Word2Vec_EN_RU.csv'
filename = 'MultiKE_Word2Vec_EN_RU_matched.csv'
filepath = folder + filename

df = prepare_df()

source = df_to_cds(df)
source2 = df_to_cds(pd.DataFrame(columns=COLUMNS))

# add interactive elements
ents = list(df['ent'])
ents.insert(0, 'None')

ent_select = Select(name='ent_select', title='Сущность', options=(ents), value=ents[0])
ent_num_slider = Slider(start=10, end=100, value=50, step=1, title='Количество')

types = list(df['type'].unique())
types.insert(0, 'All')
types_choice = MultiChoice(title='Типы', options=types, value=[types[0]])

# tap event
callback = CustomJS(args=dict(source=source), code="""
    const selector = document.getElementsByName('ent_select')[0];
    const ents = source.data.ent;

    const indices = source.selected.indices;
    if (indices.length !== 0) {
        const index = indices[0];
        const ent = ents[index];
        selector.value = ent;
    } else {
        selector.value = 'None';
    }

    var event = new Event('change');
    const cancelled = !selector.dispatchEvent(event);
""")

# create p1
select_tools = ['pan', 'wheel_zoom', 'tap', 'reset', 'save']

p1 = figure(plot_height=720, plot_width=1280, tools=select_tools, title='Векторное пространство')
p1.toolbar.active_scroll = p1.select_one(WheelZoomTool)
p1.add_tools(HoverTool(tooltips=tooltips))
p1.js_on_event('tap', callback)

p1.circle(x='x',
            y='y',
            source=source,
            color='color',
            size='sizes',
            nonselection_alpha=0.5)

# create p2
p2 = figure(plot_height=720, plot_width=1280, title='Ближайшие сущности')
p2.add_tools(HoverTool(tooltips=tooltips))

p2.circle(x='x',
            y='y',
            source=source2,
            color='color',
            size='sizes')

labels = LabelSet(x='x', y='y', text='ent', source=source2)
p2.add_layout(labels)

# create table
columns = [ TableColumn(field='ent', title='Сущность'),
            TableColumn(field='type', title='Тип'),
            TableColumn(field='distance', title='Расстояние')]
table = DataTable(source=source2, columns=columns)

# declare controls
controls = [ent_select, ent_num_slider]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

controls.insert(0, types_choice)
types_choice.on_change('value', lambda attr, old, new: update_types())

# set up layout
inputs = column(*controls, width=320)
series = column(p1, p2, table)
fields = column(row(inputs, series), sizing_mode="scale_both")

curdoc().add_root(fields)
curdoc().title = "OpenEA"