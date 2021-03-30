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
# following two lines for reading filenames from disk
from os import listdir
from os.path import isfile, join
import os
cwd = os.getcwd()
import four_step_model as fs_model


import pdb

my_path = 'data/'
onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]


prod_cons_path = 'data/prod_cons/'
prod_cons_files = [f for f in listdir(prod_cons_path) if isfile(join(prod_cons_path, f))]

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

- Η εφαρμογή αυτή δίνει την δυνατότητα στον χρήστη να προβάλλει την επιθυμητή πληροφορία από μια μεγάλη βάση δεδομένων.
- Αυτό γίνεται με χρήση φίλτρων.

'''

clarifications = '''
Διευκρινήσεις
'''


def refine_df(df):
    df = df.fillna(0)
    return df


def load_matrix(my_path, selected_matrix_fp):
    if not my_path or not selected_matrix_fp:
        print("No matrix to load")
    matrix_filepath = str(my_path) + str(selected_matrix_fp)
    my_matrix = pd.read_csv(matrix_filepath, delimiter='\t')
    return my_matrix



sample_df = []
prod_cons_df_path = ''
prod_cons_df = []
resistance_df_path = ''
resistance_df = []
image = 'url("assets/sitari-dash.png")'

nuts_names = {'Unnamed: 0':'ΠΕΡΙΦΕΡΕΙΕΣ','0':'ΑΝΑΤΟΛΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ ΚΑΙ ΘΡΑΚΗΣ','1':'ΑΤΤΙΚΗΣ','2':'ΒΟΡΕΙΟΥ ΑΙΓΑΙΟΥ','3':'ΔΥΤΙΚΗΣ ΕΛΛΑΔΑΣ','4':'ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ','5':'ΗΠΕΙΡΟΥ','6':'ΘΕΣΣΑΛΙΑΣ','7':'ΙΟΝΙΩΝ ΝΗΣΩΝ','8':'ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ','9':'ΚΡΗΤΗΣ','10':'ΝΟΤΙΟΥ ΑΙΓΑΙΟΥ','11':'ΠΕΛΟΠΟΝΝΗΣΟΥ','12':'ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ'}
nuts_list = ['ΠΕΡΙΦΕΡΕΙΕΣ', 'ΑΝΑΤΟΛΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ ΚΑΙ ΘΡΑΚΗΣ', 'ΑΤΤΙΚΗΣ', 'ΒΟΡΕΙΟΥ ΑΙΓΑΙΟΥ', 'ΔΥΤΙΚΗΣ ΕΛΛΑΔΑΣ', 'ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ', 'ΗΠΕΙΡΟΥ', 'ΘΕΣΣΑΛΙΑΣ', 'ΙΟΝΙΩΝ ΝΗΣΩΝ', 'ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ', 'ΚΡΗΤΗΣ', 'ΝΟΤΙΟΥ ΑΙΓΑΙΟΥ', 'ΠΕΛΟΠΟΝΝΗΣΟΥ', 'ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ']



chart_types = ['Γράφημα Στήλης', 'Γράφημα Πίτας']
month_dict = {0: 'Όλοι οι μήνες', 1:'Ιανουάριος', 2:'Φεβρουάριος', 3:'Μάρτιος', 4:'Απρίλιος', 5:'Μάιος', 6:'Ιούνιος', 7:'Ιούλιος', 8:'Αύγουστος', 9:'Σεπτέμβριος', 10:'Οκτώβριος', 11:'Νοέμβριος', 12:'Δεκέμβριος'}


def modify_row_titles(df, names, mod_col='Unnamed: 0'):
    # copy the names dictionary and drop the first element
    row_names = names
    row_names.pop(0)
    # replace the first column name (which is the same as the first element) with new values
    df[mod_col] = row_names
    # return the new dictionary
    return df
    


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
    dcc.Markdown(matrix_text),
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
        html.Div(id='prod-cons-input-table',  className='tableDiv'),
        # resistance table
        html.Hr(),
        html.Div(id='resistance-input-table',  className='tableDiv'),
    ]),
    html.Div([
    html.H5("Επιλογή Περιόδου"),
    dcc.Slider(id='slider',
                    min=1,
                    max=12,
                    step=1,
                    marks= month_dict,#{i: str(i) for i in range(0, 12)},
                    value=0),
    html.Div(id='output-container-slider'),
    ],  style={'backgroundColor':'#CEFFBD',
               'font-weight': 'bold',
               'fontSize' : '17px',
               'color':'#111111'}),
    html.Hr(),
    # execution button here
    html.Button('Υπολογισμός Κατανομής Μετακινήσεων', id='execution-button', n_clicks=0),
    html.Div(id='container-button-basic', className='tableDiv'),
    html.Hr(),
    html.Div([
        html.Button('Κατανομή στο Δίκτυο (networkX)', id='networkx-button', n_clicks=0),
        html.Button('Κατανομή στο Δίκτυο (ArcGIS)', id='arcgis-button', n_clicks=0),
    ], className='row',),
    html.Div(id='button-clicked-msg'),
])


@app.callback(
    Output('availability-radio-prods-cons', 'options'),
    Input('availability-radio-prods-cons', 'value'))
def set_products_options(selected_country):
    #print(selected_country)
    return [{'label': i, 'value': i} for i in prod_cons_files]


@app.callback(
    Output('prod-cons-input-table', 'children'),
    [Input('availability-radio-prods-cons', 'value'),
    Input('slider', 'value')
    ])
def set_display_table(selected_prod_cons_matrix, month_val):
    dff = load_matrix(prod_cons_path, selected_prod_cons_matrix)
    # if (month_val):
    #     dff = dff[dff[MONTH] == month_val]
    # elif month_val == 0:
    #     dff = dff
    df_temp = dff
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
    Input('slider', 'value')
    ])
def set_display_table(selected_resistance_matrix, month_val):
    dff = load_matrix(resistance_path, selected_resistance_matrix)
    # if (month_val):
    #     dff = dff[dff[MONTH] == month_val]
    # elif month_val == 0:
    #     dff = dff
    df_temp = dff
    return html.Div([
        dash_table.DataTable(
            id='main-table',
            #columns=[{'name': i, 'id': i, 'hideable':True} for i in df_temp.columns],
            columns=[{'name':val, 'id':key} for key, val in nuts_names.items()],
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
    Output('output-container-slider', 'children'),
    [Input('slider', 'value')]
)
def update_slider(value):
    if value == 0:
        return "Αποτελέσματα για όλους τους μήνες."
    return "Επιλέξατε τον {}o μήνα".format(value)



@app.callback(
    Output('container-button-basic', 'children'),
    [Input('availability-radio-prods-cons', 'value'),
     Input('availability-radio-resistance', 'value'),
     Input('execution-button', 'n_clicks')])
def update_output(prod_cons_matrix, resistance_matrix, click_value):
    prod_cons_input = str(prod_cons_path) + str(prod_cons_matrix)
    resistance_input = str(resistance_path) + str(resistance_matrix)
    if click_value > 0:
        results = fs_model.four_step_model(prod_cons_input, resistance_input, 0.1)
        dff = load_matrix(results_path, results_filepath)
        df_temp = dff
        (styles, legend) = discrete_background_color_bins(df_temp, n_bins=7, columns='all')
        df_temp = modify_row_titles(df_temp, nuts_list)
        return html.Div([
            html.Div(legend, style={'float': 'right'}),
    dash_table.DataTable(
        data=df_temp.to_dict('records'),
        sort_action='native',
        columns=[{'name':val, 'id':key} for key, val in nuts_names.items()],
        #columns=[{'name': i, 'id': i} for i in df_temp.columns],
        style_data_conditional=styles
    ),
        ])
        # return html.Div([
        #     dash_table.DataTable(
        #         id='button-table',
        #         columns=[{'name': i, 'id': i, 'hideable':True} for i in df_temp.columns],
        #         data=df_temp.to_dict('rows'),
        #         editable=True,
        #         filter_action='native',
        #         sort_action='native',
        #         sort_mode="multi",
        #         column_selectable="single",
        #         row_selectable="multi",
        #         row_deletable=True,
        #         selected_columns=[],
        #         selected_rows=[],
        #         hidden_columns=['LastDayWeek', 'week'],
        #         #  page_action="native",
        #         #  page_current= 0,
        #         page_size= 15,
        #         style_table={
        #             'maxHeight': '50%',
        #             'overflowY': 'scroll',
        #             'width': '100%',
        #             'minWidth': '10%',
        #         },
        #         style_header={'backgroundColor': 'rgb(200,200,200)', 'width':'auto'},
        #         style_cell={'backgroundColor': 'rgb(230,230,230)','color': 'black','height': 'auto','minWidth': '100px', 'width': '150px', 'maxWidth': '180px','overflow': 'hidden', 'textOverflow': 'ellipsis', },#minWidth': '0px', 'maxWidth': '180px', 'whiteSpace': 'normal'},
        #         #style_cell={'minWidth': '120px', 'width': '150px', 'maxWidth': '180px'},
        #         style_data={'whiteSpace': 'auto','height': 'auto','width': 'auto'},
        #         tooltip_data=[
        #         {
        #             column: {'value': str(value), 'type': 'markdown'}
        #             for column, value in row.items()
        #         } for row in df_temp.to_dict('records')
        #         ],
        #         tooltip_header={i: i for i in df_temp.columns},
        # tooltip_duration=None
        #     )
        # ])
    else:
        return html.Div("Προς υπολογισμό αποτελεσμάτων.")


@app.callback(Output('button-clicked-msg', 'children'),
              Input('networkx-button', 'n_clicks'),
              Input('arcgis-button', 'n_clicks'))
def displayClick(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'networkx-button' in changed_id:
        msg = 'Networkx clicked'
    elif 'arcgis-button' in changed_id:
        msg = 'ArcGIS button clicked'
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div(msg)



if __name__ == '__main__':
    app.run_server(debug=True)