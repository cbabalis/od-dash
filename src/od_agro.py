import random
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash_table import DataTable
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import base64
import plotly.graph_objects as go
# flask
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
# following two lines for reading filenames from disk
from os import listdir
from os.path import isfile, join
import os
cwd = os.getcwd()
import four_step_model_updated as fs_model
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
import pdb


my_path = 'data/'
onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
download_df = [] # file for downloading

prod_cons_path = 'data/prod_cons/'
#prod_cons_files = uploaded_files(prod_cons_path) # [f for f in listdir(prod_cons_path) if isfile(join(prod_cons_path, f))]

resistance_path = 'data/resistance/'
resistance_files = [f for f in listdir(resistance_path) if isfile(join(resistance_path, f))]

results_path = 'results/'
results_filepath = 'output.csv'

matrix_text = '''
#### OD Matrix κείμενο εδώ
'''

help_text = '''
ΕΠΕΞΗΓΗΣΕΙΣ ΤΗΣ ΕΦΑΡΜΟΓΗΣ

*Οδηγίες χρήσης:*

- Η εφαρμογή αυτή δίνει την δυνατότητα στον χρήστη να δημιουργήσει πίνακα OD με είσοδο πίνακες με παραγωγές, καταναλώσεις και μητρώα αντίστασης μετακίνησης.

1. Ο χρήστης επιλέγει τον πίνακα παραγωγών και καταναλώσεων από το πεδίο ΣΤΟΙΧΕΙΑ ΠΑΡΑΓΩΓΗΣ-ΚΑΤΑΝΑΛΩΣΗΣ
2. Ο χρήστης επιλέγει το μητρώο αντίστασης μετακινήσεων από το αντίστοιχο πεδίο (ΜΗΤΡΩΟ ΑΝΤΙΣΤΑΣΗΣ ΜΕΤΑΚΙΝΗΣΕΩΝ).
3. Πατώντας το κουμπί "ΥΠΟΛΟΓΙΣΜΟΣ ΚΑΤΑΝΟΜΗΣ ΜΕΤΑΚΙΝΗΣΕΩΝ" ο χρήστης παίρνει τον ζητούμενο OD πίνακα.
'''

clarifications = '''
Διευκρινήσεις
'''


def refine_df(df):
    df = df.fillna(0)
    return df


def load_matrix(my_path, selected_matrix_fp, delim='\t'):
    if not my_path or not selected_matrix_fp:
        print("No matrix to load")
    matrix_filepath = str(my_path) + str(selected_matrix_fp)
    my_matrix = pd.read_csv(matrix_filepath, delimiter=delim)
    return my_matrix


sample_df = []
prod_cons_df_path = ''
prod_cons_df = []
resistance_df_path = ''
resistance_df = []
image = 'url("assets/sitari-dash.png")'

gis_img = cwd + '/src/assets/GIS1b.png' # replace with your own image
encoded_image = base64.b64encode(open(gis_img, 'rb').read())

nuts_names = {'Unnamed: 0':'ΠΕΡΙΦΕΡΕΙΕΣ','0':'ΑΝΑΤΟΛΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ ΚΑΙ ΘΡΑΚΗΣ','1':'ΑΤΤΙΚΗΣ','2':'ΒΟΡΕΙΟΥ ΑΙΓΑΙΟΥ','3':'ΔΥΤΙΚΗΣ ΕΛΛΑΔΑΣ','4':'ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ','5':'ΗΠΕΙΡΟΥ','6':'ΘΕΣΣΑΛΙΑΣ','7':'ΙΟΝΙΩΝ ΝΗΣΩΝ','8':'ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ','9':'ΚΡΗΤΗΣ','10':'ΝΟΤΙΟΥ ΑΙΓΑΙΟΥ','11':'ΠΕΛΟΠΟΝΝΗΣΟΥ','12':'ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ'}
nuts_list = ['ΠΕΡΙΦΕΡΕΙΕΣ', 'ΑΝΑΤΟΛΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ ΚΑΙ ΘΡΑΚΗΣ', 'ΑΤΤΙΚΗΣ', 'ΒΟΡΕΙΟΥ ΑΙΓΑΙΟΥ', 'ΔΥΤΙΚΗΣ ΕΛΛΑΔΑΣ', 'ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ', 'ΗΠΕΙΡΟΥ', 'ΘΕΣΣΑΛΙΑΣ', 'ΙΟΝΙΩΝ ΝΗΣΩΝ', 'ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ', 'ΚΡΗΤΗΣ', 'ΝΟΤΙΟΥ ΑΙΓΑΙΟΥ', 'ΠΕΛΟΠΟΝΝΗΣΟΥ', 'ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ']

resistance_title_names = []


chart_types = ['Γράφημα Στήλης', 'Γράφημα Πίτας']
month_dict = {0: 'Όλοι οι μήνες', 1:'Ιανουάριος', 2:'Φεβρουάριος', 3:'Μάρτιος', 4:'Απρίλιος', 5:'Μάιος', 6:'Ιούνιος', 7:'Ιούλιος', 8:'Αύγουστος', 9:'Σεπτέμβριος', 10:'Οκτώβριος', 11:'Νοέμβριος', 12:'Δεκέμβριος'}


def modify_row_titles(df, names, mod_col='Unnamed: 0'):
    # copy the names dictionary and drop the first element
    row_names = [elem for elem in names]
    if 'Unnamed: 0' in row_names:
        del row_names['Unnamed: 0']
    # replace the first column name (which is the same as the first element) with new values
    df[mod_col] = row_names
    # return the new dictionary
    return df


def _get_od_column_names(resistance_title_names, nuts_names, df):
    if not resistance_title_names or len(df) == 13:
        return [{'name':val, 'id':key} for key, val in nuts_names.items()]
    else:
        ids_list = [id for id in range(len(resistance_title_names))]
        return [{'name':val, 'id':key} for key, val in zip(ids_list, resistance_title_names)]
    


def discrete_background_color_bins(df, n_bins=9, columns='all'):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'
        #backgroundColor = colorlover.scales[str(n_bins+4)]['div']['RdYlGn'][2:-2][i - 1]
        #color = 'black'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Μεταφορές μεταξύ Περιοχών (Μητρώο Προέλευσης/Προορισμού)",  style={'textAlign':'center'}),
    html.Hr(),
    # text here
    html.Div([
    #dcc.Markdown(matrix_text),
    dcc.ConfirmDialogProvider(children=html.Button(
            'Οδηγίες Χρήσης',
            style={'float': 'right','margin': 'auto'}
        ),
        id='danger-danger-provider',
        message=help_text,
    ),
    html.Div(id='output-provider')
    ],
             className='row'),
    # filters here
    html.Div([
        html.Div([
            html.Div([
                html.Label("ΣΤΟΙΧΕΙΑ ΠΑΡΑΓΩΓΗΣ-ΚΑΤΑΝΑΛΩΣΗΣ",
                    style={'font-weight': 'bold',
                            'fontSize' : '17px',
                            'margin-left':'auto',
                            'margin-right':'auto',
                            'display':'block'}),
                dcc.Dropdown(id='availability-radio-prods-cons',
                            style={"display": "block",
                    "margin-left": "auto",
                    "margin-right": "auto",
                    # "width":"60%"
                    }), # style solution here: https://stackoverflow.com/questions/51193845/moving-objects-bar-chart-using-dash-python
                ## radio button to select by what
                 html.Label("ΕΠΙΠΕΔΟ ΓΕΩΓΡΑΦΙΚΗΣ ΑΝΑΛΥΣΗΣ",
                    style={'font-weight': 'bold',
                            'fontSize' : '17px',
                            'margin-left':'auto',
                            'margin-right':'auto',
                            'display':'block'}),
                dcc.Dropdown(id='region-selection',
                            style={"display": "block",
                    "margin-left": "auto",
                    "margin-right": "auto",
                    # "width":"60%"
                    }),
                ## end of radio button to select by what
            ], className='four columns'),
            html.Div([
                html.Label("ΜΗΤΡΩΟ ΑΝΤΙΣΤΑΣΗΣ ΜΕΤΑΚΙΝΗΣΕΩΝ",
                        style={'font-weight': 'bold',
                                'fontSize' : '17px'}),
                dcc.Dropdown(id='availability-radio-resistance',
                            style={"display": "block",
                    "margin-left": "auto",
                    "margin-right": "auto",
                    # "width":"60%"
                    }),
                dcc.ConfirmDialogProvider(children=html.Button(
            'Διευκρινησεις',
            style={'float': 'right','margin': 'auto','background-color':'white'}
        ),
        id='clarifications',
        message=clarifications,
    ),
                html.Div(id='output-provider-clarifications'),
            ], className='four columns'),
        ], className='row',
            style= {'padding-left' : '50px',
                    'padding-right': '50px'}), # closes the div for first line (matrix and year)
        html.Hr(),
    ],style = {'background-image':image,
                'background-size':'cover',
                'background-position':'right'}),
    # tables here
    html.Div([
        # productions table
        html.Label('Παραγωγή και κατανάλωση ανά γεωγραφική ενότητα'),
        html.Div(id='prod-cons-input-table',  className='tableDiv'),
        # resistance table
        html.Hr(),
        html.Label('Mητρώο αντίστασης μετακινήσεων μεταξύ γεωγραφικών ενοτήτων'),
        html.Div(id='resistance-input-table',  className='tableDiv'),
        # slider for choosing percentage of internal movements
        html.Div([
            html.Label('Ποσοστό εσωτερικής Κατανάλωσης στους νομούς'),
            dcc.Slider(id='internal-movement-slider',
                min=10,
                max=70,
                value=35,
                marks={
                    10: {'label': '10%', 'style': {'color': '#77b0b1'}},
                    35: {'label': '35%'},
                    50: {'label': '50%'},
                    75: {'label': '75%', 'style': {'color': '#f50'}}
                }
            ),
        ], style={'width': '19%', 'display': 'inline-block', 'vertical-align': 'middle'}),
    ]),
    html.Hr(),
    # execution button here
    html.Div([
    html.Button('Υπολογισμός Κατανομής Μετακινήσεων', id='execution-button',n_clicks=0),
    ], style={'margin-bottom': '10px',
              'textAlign':'center',
              'width': '220px',
              'margin':'auto'}),
    html.Div(id='container-button-basic', className='tableDiv'),
    html.Hr(),
    html.Div(
        [
            html.Button("Κατεβασμα δεδομένων (CSV)", id="btn_csv"),
            Download(id="download-dataframe-csv"),
        ],
    ),
    html.Div([
        html.Button('Κατανομή στο Δίκτυο (networkX)', id='networkx-button', n_clicks=0),
        html.Button('Κατανομή στο Δίκτυο (ArcGIS)', id='arcgis-button', n_clicks=0),
    ], className='row', style={'margin-bottom': '10px',
              'textAlign':'center',
              'width': '1020px',
              'margin':'auto'}),
    html.Div(id='button-clicked-msg'),
])


@app.callback(
    Output('availability-radio-prods-cons', 'options'),
    Input('availability-radio-prods-cons', 'value'))
def set_products_options(selected_country):
    #print(selected_country)
    prod_cons_files = [f for f in listdir(prod_cons_path) if isfile(join(prod_cons_path, f))] #uploaded_files(prod_cons_path)
    return [{'label': i, 'value': i} for i in prod_cons_files]


@app.callback(
    Output('region-selection', 'options'),
    Input('region-selection', 'value'))
def set_region_group_by_options(selected_country):
    region_choices = ['ΠΕΡΙΦΕΡΕΙΑ', 'ΠΕΡΙΦΕΡΕΙΑΚΕΣ ΕΝΟΤΗΤΕΣ']
    return [{'label': i, 'value': i} for i in region_choices]


@app.callback(
    Output('prod-cons-input-table', 'children'),
    [Input('availability-radio-prods-cons', 'value'),
     Input('region-selection', 'value'),
    ])
def set_display_table(selected_prod_cons_matrix, reg_sel):
    dff = load_matrix(prod_cons_path, selected_prod_cons_matrix)
    df_temp = dff.round(2)
    # assign names to a list
    global resistance_title_names
    if reg_sel in df_temp.columns:
        resistance_title_names = df_temp[reg_sel].tolist()
    # if 'ΠΕΡΙΦΕΡΕΙΑΚΗ ΕΝΟΤΗΤΑ' in df_temp.columns:
    #     resistance_title_names = df_temp['ΠΕΡΙΦΕΡΕΙΑΚΗ ΕΝΟΤΗΤΑ'].tolist()
    # elif 'ΠΕΡΙΦΕΡΕΙΑ' in df_temp.columns:
    #     resistance_title_names = df_temp['ΠΕΡΙΦΕΡΕΙΑ'].tolist()
    assert resistance_title_names, "resistance title names length is %d" % len(resistance_title_names) # https://stackoverflow.com/questions/1308607/python-assert-improved-introspection-of-failure
    return html.Div([
        dash_table.DataTable(
            id='main-table',
            columns=[{'name': i, 'id': i, 'hideable':True} for i in df_temp.columns],
             data=df_temp.to_dict('rows'),
             editable=True,
             filter_action='native',
             sort_action='native',
             sort_mode="multi",
             column_selectable="single",
             row_selectable="multi",
             row_deletable=True,
             selected_columns=[],
             selected_rows=[],
             hidden_columns=['LastDayWeek', 'week'],
            #  page_action="native",
            #  page_current= 0,
             page_size= 5,
             style_table={
                'maxHeight': '50%',
                'overflowY': 'scroll',
                'width': '100%',
                'minWidth': '10%',
            },
            style_header={'backgroundColor': 'rgb(200,200,200)', 'width':'auto'},
            style_cell={'backgroundColor': 'rgb(230,230,230)','color': 'black','height': 'auto','minWidth': '100px', 'width': '150px', 'maxWidth': '180px','overflow': 'hidden', 'textOverflow': 'ellipsis', },#minWidth': '0px', 'maxWidth': '180px', 'whiteSpace': 'normal'},
            #style_cell={'minWidth': '120px', 'width': '150px', 'maxWidth': '180px'},
            style_data={'whiteSpace': 'auto','height': 'auto','width': 'auto'},
            tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in df_temp.to_dict('records')
            ],
            tooltip_header={i: i for i in df_temp.columns},
    tooltip_duration=None
        )
    ])


@app.callback(
    Output('availability-radio-resistance', 'options'),
    Input('availability-radio-resistance', 'value'))
def set_products_options(selected_country):
    return [{'label': i, 'value': i} for i in resistance_files]



@app.callback(
    Output('resistance-input-table', 'children'),
    [Input('availability-radio-resistance', 'value'),
    ])
def set_display_table(selected_resistance_matrix):
    dff = load_matrix(resistance_path, selected_resistance_matrix)
    # if (month_val):
    #     dff = dff[dff[MONTH] == month_val]
    # elif month_val == 0:
    #     dff = dff
    df_temp = dff
    nuts_names_temp = nuts_names
    #if  nuts_names_temp['Unnamed: 0']:
    if 'Unnamed: 0' in nuts_names_temp.keys():
        del nuts_names_temp['Unnamed: 0']
    # get names of columns of OD matrix (it is dependent on the prod_cons_ file)
    od_cols = _get_od_column_names(resistance_title_names, nuts_names_temp, df_temp)
    assert len(od_cols) == len(df_temp), "titles' length does not match dataframe's length"
    df_temp.columns = od_cols
    return html.Div([
        dash_table.DataTable(
            id='resistance-table',
            #columns=[{'name': i, 'id': i, 'hideable':True} for i in df_temp.columns],
            #columns= [{'name':val, 'id':val, 'hideable':True} for val in resistance_title_names], #[{'name':val, 'id':key} for key, val in nuts_names_temp.items()],
            columns = od_cols,
            data=df_temp.to_dict('records'),
            editable=True,
            filter_action='native',
            sort_action='native',
            sort_mode="multi",
            column_selectable="single",
            #row_selectable="multi",
            row_deletable=True,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 5,
            style_table={
                'maxHeight': '50%',
                'overflowY': 'scroll',
                'width': '100%',
                'minWidth': '10%',
            },
            style_header={'backgroundColor': 'rgb(200,200,200)', 'width':'auto'},
            style_cell={'backgroundColor': 'rgb(230,230,230)','color': 'black','height': 'auto','minWidth': '100px', 'width': '150px', 'maxWidth': '180px','overflow': 'hidden', 'textOverflow': 'ellipsis', },#minWidth': '0px', 'maxWidth': '180px', 'whiteSpace': 'normal'},
            #style_cell={'minWidth': '120px', 'width': '150px', 'maxWidth': '180px'},
            style_data={'whiteSpace': 'auto','height': 'auto','width': 'auto'},
            tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in df_temp.to_dict('records')
            ],
            tooltip_header={i: i for i in df_temp.columns},
    tooltip_duration=None
        )
    ])



@app.callback(Output('output-provider', 'children'),
              Input('danger-danger-provider', 'submit_n_clicks'))
def update_output(submit_n_clicks):
    """ documentation: https://dash.plotly.com/dash-core-components/confirmdialogprovider"""
    if not submit_n_clicks:
        return ''
    return """
        Ευχαριστούμε που χρησιμοποιήσατε τις οδηγίες.
    """


@app.callback(Output('output-provider-clarifications', 'children'),
              Input('clarifications', 'submit_n_clicks'))
def update_output(submit_n_clicks):
    """ documentation: https://dash.plotly.com/dash-core-components/confirmdialogprovider"""
    if not submit_n_clicks:
        return ''
    return ''


@app.callback(
    Output('container-button-basic', 'children'),
    [Input('execution-button', 'n_clicks')],
    [State('availability-radio-prods-cons', 'value'),
     State('availability-radio-resistance', 'value'),
     State('region-selection', 'value'),
     State('internal-movement-slider', 'value')])
def update_output(click_value, prod_cons_matrix, resistance_matrix, region_lvl, internal_pcnt):
    prod_cons_input = str(prod_cons_path) + str(prod_cons_matrix)
    resistance_input = str(resistance_path) + str(resistance_matrix)
    if not click_value:
        return dash.no_update
    #elif click_value > 0:
    results = fs_model.four_step_model(prod_cons_input, resistance_input, 1, internal_pcnt, group_by_col=region_lvl)
    dff = load_matrix(results_path, results_filepath)
    df_temp = dff
    (styles, legend) = discrete_background_color_bins(df_temp, n_bins=7, columns='all')
    # create results columns' names
    results_cols = [{'name': i, 'id': i, 'hideable':True} for i in df_temp.columns] #_get_od_column_names(resistance_title_names, nuts_names, df_temp)
    # following five lines are about to create a file for downloading the results
    results_index = df_temp.columns.tolist()
    results_index.pop(0)
    df_temp['Unnamed: 0'] = results_index
    global download_df
    download_df = df_temp
    return html.Div([
        html.Div(legend, style={'float': 'right'}),
        dash_table.DataTable(
            data=df_temp.to_dict('records'),
            sort_action='native',
            columns= results_cols,
            page_action="native",
            page_current= 0,
            page_size= 15,
            style_table={
                'maxHeight': '50%',
                'overflowY': 'scroll',
                'width': '100%',
                'minWidth': '10%',
            },
            style_header={'backgroundColor': 'rgb(200,200,200)', 'width':'auto'},
            editable=True,
            filter_action='native',
            row_selectable="multi",
            style_data_conditional=styles
        ),
    ])
    #else:
    #    return html.Div("Για αποτελέσματα πατήστε το κουμπί 'Υπολογισμός Κατανομής Μετακινήσεων'.")


@app.callback(Output('button-clicked-msg', 'children'),
              Input('networkx-button', 'n_clicks'),
              Input('arcgis-button', 'n_clicks'))
def displayClick(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'networkx-button' in changed_id:
        msg = 'Networkx clicked'
    elif 'arcgis-button' in changed_id:
        msg = 'ArcGIS button clicked'
        msg = html.Img(src=app.get_asset_url('GIS1b.png'))
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div(msg)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    [State('availability-radio-prods-cons', 'value'),
     State('region-selection', 'value')],
    prevent_initial_call=True,
)
def func(n_clicks, prod_cons_matrix, region_lvl):
    temp_dff = load_matrix(str(prod_cons_path), str(prod_cons_matrix))
    return send_data_frame(download_df.to_csv, "mydf.csv") # dash_extensions.snippets: send_data_frame
    # temp_dff = load_matrix(str(prod_cons_path), str(prod_cons_matrix))
    # resistance_title_cols = temp_dff[region_lvl].unique()
    # resistance_title_cols = resistance_title_cols.tolist()
    # assert len(resistance_title_cols) == len(download_df), "lengths are not the same: %d, %d" % (len(resistance_title_cols), len(download_df))
    # if 'Unnamed: 0' in download_df:
    #     del download_df['Unnamed: 0']
    # download_df.columns = resistance_title_cols
    # download_df.insert(0, 'Unnamed: 0', resistance_title_cols)
    # return send_data_frame(download_df.to_csv, "mydf.csv") # dash_extensions.snippets: send_data_frame



if __name__ == "__main__":
    app.run_server(debug=False, port=8056) # host='147.102.154.65', 