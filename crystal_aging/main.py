from functools import lru_cache

from os import listdir
from os.path import dirname, join

import re
import math
import numpy as np
import pandas as pd
from lmfit import Model

from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox, layout
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Div, Legend, Legend
from bokeh.models.widgets import DataTable, Select, TableColumn, Button, CheckboxGroup, TextInput
from bokeh.plotting import figure, show

from io import StringIO
import base64

def default_model(x, a, b):
    return a + b*np.log(x)

def load_data(attr, old, new):
    global df
    df = pd.read_csv(join(dirname(__file__), 'data/'+data_select.value), names=['Day', 'Projection', 'Measurement'])
    aDf = pd.DataFrame(data={'x' : df.Day, 'y' : df.Measurement})
    a_source.data = a_source.from_df(aDf)
    bDf = pd.DataFrame(data={'x' : df.Day, 'y' : df.Projection})
    b_source.data = b_source.from_df(bDf)
    model = Model(default_model)
    result = model.fit(df.Measurement, x=df.Day, a=1, b=1)
    fit = result.best_fit
    cDf = pd.DataFrame(data={'x' : df.Day, 'y' : fit})
    c_source.data = c_source.from_df(cDf)
    dDf = pd.DataFrame(data={'a' : result.best_values['a'], 'b' : result.best_values['b']}, index=[0])
    source_data_table.data = source_data_table.from_df(dDf)
    proj_resid = np.sum((df.Projection-df.Measurement)**2)
    fit_resid = np.sum((df.Projection-fit)**2)
    df_residual = pd.DataFrame(data={'Projection Residual' : [proj_resid], 'Fit Residual' : [fit_resid]})
    source_residual_table.data = source_residual_table.from_df(df_residual)
    p.title.text = "Part #"+str(data_select.value.split('.')[0])
    update()

def selection_range(attr, old, new):
    selected_range = np.sort(a_source.selected.indices)
    if 0 in fit_select.active:
        bDf = pd.DataFrame(data={'x' : df.iloc[selected_range].Day, 'y' : df.iloc[selected_range].Projection})
        b_source.data = b_source.from_df(bDf)
        proj_resid = np.sum((df.iloc[selected_range].Projection-df.iloc[selected_range].Measurement)**2)
    if 1 in fit_select.active:
        model = Model(default_model)
        result = model.fit(df.iloc[selected_range].Measurement, x=df.iloc[selected_range].Day, a=1, b=1)
        fit = result.best_fit
        cDf = pd.DataFrame(data={'x' : df.iloc[selected_range].Day, 'y' : fit})
        c_source.data = c_source.from_df(cDf)
        dDf = pd.DataFrame(data={'a' : result.best_values['a'], 'b' : result.best_values['b']}, index=[0])
        source_data_table.data = source_data_table.from_df(dDf)
        fit_resid = np.sum((df.iloc[selected_range].Projection-fit)**2)
    df_residual = pd.DataFrame(data={'Projection Residual' : [proj_resid], 'Fit Residual' : [fit_resid]})
    source_residual_table.data = source_residual_table.from_df(df_residual)
    update()

def update_fits(attr, old, new):
    if 0 in fit_select.active:
        bDf = pd.DataFrame(data={'x' : df.Day, 'y' : df.Projection})
        b_source.data = b_source.from_df(bDf)
        proj_resid = np.sum((df.Projection-df.Measurement)**2)
        source_residual_table.data['Projection Residual'] = [proj_resid]
    if 1 in fit_select.active:
        model = Model(default_model)
        result = model.fit(df.Measurement, x=df.Day, a=1, b=1)
        fit = result.best_fit
        cDf = pd.DataFrame(data={'x' : df.Day, 'y' : fit})
        c_source.data = c_source.from_df(cDf)
        dDf = pd.DataFrame(data={'a' : result.best_values['a'], 'b' : result.best_values['b']}, index=[0])
        source_data_table.data = source_data_table.from_df(dDf)
        fit_resid = np.sum((df.Projection-fit)**2)
        source_residual_table.data['Fit Residual'] = [fit_resid]
    update()

def update():
    if 0 not in fit_select.active:
        bDf = pd.DataFrame(data={'x' : [], 'y' : []})
        b_source.data = b_source.from_df(bDf)
        proj_resid = '-'
        source_residual_table.data['Projection Residual'] = [proj_resid]
    if 1 not in fit_select.active:
        cDf = pd.DataFrame(data={'x' : [], 'y' : []})
        c_source.data = c_source.from_df(cDf)
        dDf = pd.DataFrame(data={'a' : '-', 'b' : '-'}, index=0)
        source_data.data = source_data.from_df(dDf)
        fit_resid = '-'
        source_residual_table.data['Fit Residual'] = [fit_resid]

sample_data = listdir(join(dirname(__file__), 'data'))
parts = []
for sd in sample_data:
    parts.append(sd.split('.')[0])

data_select = Select(title='Select Sample Data File', options=sample_data, value=sample_data[0])
data_select.on_change('value', load_data)

fit_select = CheckboxGroup(labels=['Plot Projection', 'Plot Fit'], active=[0, 1])
fit_select.on_change('active', update_fits)

df = pd.read_csv(join(dirname(__file__), 'data/'+data_select.value), names=['Day', 'Projection', 'Measurement'])

tools_model = 'xbox_select,wheel_zoom,pan,reset,save'
p = figure(title="Part #"+str(parts[0]), x_axis_label="Days", y_axis_label="ppb", plot_width=600, plot_height=600, tools=tools_model,
          toolbar_location='left')

aDf = pd.DataFrame(data={'x' : df.Day, 'y' : df.Measurement})
a_source = ColumnDataSource(aDf)
bDf = pd.DataFrame(data={'x' : df.Day, 'y' : df.Projection})
b_source = ColumnDataSource(bDf)
model = Model(default_model)
result = model.fit(df.Measurement, x=df.Day, a=1, b=1)
fit = result.best_fit
cDf = pd.DataFrame(data={'x' : df.Day, 'y' : fit})
c_source = ColumnDataSource(cDf)

a = p.circle('x', 'y', size=8, source=a_source, color='grey', alpha=0.6)
b = p.line('x', 'y', source=b_source, line_width=3, color='blue', alpha=0.4)
c = p.line('x', 'y', source=c_source, line_width=3, color='red', alpha=0.4)

legend = Legend(items=[
    ("Measurement", [a]),
    ("Projection", [b]),
    ("y = a + b*ln(x)", [c])
])

source_data = {'a' : [result.best_values['a']], 'b' : [result.best_values['b']]}
source_data_table = ColumnDataSource(source_data)
columns = [
    TableColumn(field='a', title='a'),
    TableColumn(field='b', title='b')
]
data_table = DataTable(source=source_data_table, columns=columns, width=300, height=50)

proj_resid = np.sum((df.Projection-df.Measurement)**2)
fit_resid = np.sum((df.Projection-fit)**2)

source_residual = {'Projection Residual' : [proj_resid], 'Fit Residual' : [fit_resid]}
source_residual_table = ColumnDataSource(source_residual)
rcolumns = [
    TableColumn(field='Projection Residual', title='Projection Residual'),
    TableColumn(field='Fit Residual', title='Fit Residual')
]
residual_table = DataTable(source=source_residual_table, columns=rcolumns, width=300, height=50)

a_source.on_change('selected', selection_range)

p.add_layout(legend, 'right')

sizing_mode = 'fixed'

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=1000)

l = layout([
    [desc],
    [row(widgetbox(data_select, fit_select), p, column(data_table, residual_table))]
], sizing_mode=sizing_mode)

update()

curdoc().add_root(l)
curdoc().title = 'crystal_aging'
