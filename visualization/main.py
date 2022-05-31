import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool, WheelZoomTool
from bokeh.models import Select
from bokeh.models.callbacks import CustomJS


# Load data
folder = 'data\\'
filename = 'RDGCN_EN_RU_15K_V1_labse.csv'
filepath = folder + filename
df_main = pd.read_csv(filepath)

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], ent1_id=[], ent2_id=[], ent1=[], ent2=[], lang=[], type_=[], color=[], size=[], distance=[]))


# Form buttons
types = list(df_main['type'].unique())
types = sorted(types)
types.insert(0, 'All')
select_type = Select(name='select_type', title='Тип', options=types, value=types[0])

ids = list(df_main['ent1_id'])
ids = list(map(str, ids))
ids.insert(0, '-1')
select_id = Select(name='select_id', title='Номер', options=ids, value=ids[0], visible=False)


# Form screens
select_tools = ['pan', 'wheel_zoom', 'tap', 'reset', 'save']
tooltips = [('Entity', '@ent1' + ' (@lang)'), ('Type', '@type_')]

p1 = figure(plot_height=720, plot_width=1280, tools=select_tools, title='Векторное пространство')
p1.toolbar.active_scroll = p1.select_one(WheelZoomTool)
p1.add_tools(HoverTool(tooltips=tooltips))

callback = CustomJS(args=dict(source=source), code="""
    const selector = document.getElementsByName('select_id')[0];
    const indices = source.selected.indices;
    if (indices.length !== 0) {
        selector.value = indices[0];
    } else {
        selector.value = -1;
    }
    var event = new Event('change');
    const cancelled = !selector.dispatchEvent(event);
""")
p1.js_on_event('tap', callback)

p1.circle(x='x',
          y='y',
          source=source,
          color='color',
          size='size',
          nonselection_alpha=0.5)


def set_colors(df):
    languages = list(df['lang'])
    colors = list(map(lambda x: 'blue' if x == 'en' else 'red', languages))
    df['color'] = colors
    return df


def set_params(df):
    df = set_colors(df)
    df['size'] = 5
    df['distance'] = 0
    return df


def get_color(index):
    color = 'blue'
    if index % 2 != 0:
        color = 'red'
    return color


def emphasize_pair(df, df_ids, id1):
    row = df.loc[df['ent1_id'] == id1]
    id2 = row['ent2_id'].values[0]
    pair = [id1, id2]

    colors = list(map(lambda x: get_color(x) if x in pair else 'lightgray', df_ids))
    df['color'] = colors
    sizes = list(map(lambda x: 10 if x in pair else 5, df_ids))
    df['size'] = sizes
    return df


def get_data():
    current_type = select_type.value
    df = df_main.copy()
    df = set_params(df)
    if current_type != 'All':
        df = df.loc[df['type'] == current_type]

    current_id = int(select_id.value)
    df_ids = list(df['ent1_id'])
    if current_id != -1:
        if current_id < len(df_ids):
            if len(df) < len(df_main):
                current_id = df_ids[current_id]
            df = emphasize_pair(df, df_ids, current_id)
        else:
            source.selected.indices = []

    return df


def update():
    df = get_data()
    source.data = dict(
        x=df['x'], y=df['y'], 
        ent1_id=df['ent1_id'], ent2_id=df['ent2_id'], 
        ent1=df['ent1'], ent2=df['ent2'], 
        lang=df['lang'], type_=df['type'], 
        color=df['color'], size=df['size'], 
        distance=df['distance']
    )


# declare controls
controls = [select_type, select_id]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())


# set up layout
inputs = column(*controls, width=320)
series = column(p1)
fields = column(row(inputs, series), sizing_mode="scale_both")

update()  # initial load of the data

curdoc().add_root(fields)
curdoc().title = filename