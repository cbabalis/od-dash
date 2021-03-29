import random
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_table import DataTable
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
# following two lines for reading filenames from disk
from os import listdir
from os.path import isfile, join
import os


import pdb

my_path = 'data/'
onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]


prod_cons_path = 'data/prod_cons/'
prod_cons_files = [f for f in listdir(prod_cons_path) if isfile(join(prod_cons_path, f))]

resistance_path = 'data/antist/'
resistance_files = [f for f in listdir(resistance_path) if isfile(join(resistance_path, f))]

matrix_text = '''
#### OD Matrix κείμενο εδώ
'''

help_text = '''
ΕΠΕΞΗΓΗΣΕΙΣ ΤΗΣ ΕΦΑΡΜΟΓΗΣ

*Οδηγίες χρήσης:*

- Η εφαρμογή αυτή δίνει την δυνατότητα στον χρήστη να προβάλλει την επιθυμητή πληροφορία από μια μεγάλη βάση δεδομένων.
- Αυτό γίνεται με χρήση φίλτρων.

'''


def refine_df(df):
    df = df.fillna(0)
    return df


def load_matrix(my_path, selected_matrix_fp, my_matrix=''):
    if not my_matrix:
        print("No matrix to load")
    matrix_filepath = my_path + selected_matrix
    my_matrix = pd.read_csv(matrix_filepath, delimiter='\t')
    return my_matrix


sample_df = []
prod_cons_df = []
resistance_df = []
# doc for image: https://community.plotly.com/t/background-image/21199/5
#image = 'url(http://147.102.154.65/enirisst/images/ampeli-dash.png)'
image = 'url("assets/ampeli-dash.png")'


chart_types = ['Γράφημα Στήλης', 'Γράφημα Πίτας']
month_dict = {0: 'Όλοι οι μήνες', 1:'Ιανουάριος', 2:'Φεβρουάριος', 3:'Μάρτιος', 4:'Απρίλιος', 5:'Μάιος', 6:'Ιούνιος', 7:'Ιούλιος', 8:'Αύγουστος', 9:'Σεπτέμβριος', 10:'Οκτώβριος', 11:'Νοέμβριος', 12:'Δεκέμβριος'}


def get_col_rows_data(selected_country, selected_city, sample_df):
    if selected_country == '':
        df_temp = sample_df
    elif (isinstance(selected_city, str)):
        df_temp = sample_df[sample_df[selected_country] == selected_city]
    else:
        df_temp= sample_df[sample_df[selected_country].isin(selected_city)]
    return df_temp


def get_bar_figure(dff, x_col, y_col, col_sum):
    fig = px.bar(dff, x=x_col, y=y_col, color=col_sum)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest',
        #title= u'Διάγραμμα μεταβλητών {} και {}'.format(x_col,y_col),
        font=dict(
        family="Courier New, monospace",
        size=15,
        color="RebeccaPurple"
    ))
    
    fig.update_traces( textposition='auto')

    fig.update_xaxes(title=y_col)

    fig.update_yaxes(title=col_sum)
    
    return fig


def get_pie_figure(dff, x_col, col_sum, y_col):
    fig = px.pie(dff, values=col_sum, names=y_col)
    fig.update_traces(textposition='inside', textinfo='percent', hoverinfo='label+value', textfont_size=20)
    fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide',
        font=dict(
        family="Courier New, monospace",
        size=15,
        color="RebeccaPurple"
    ))
    return fig



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Βάση Δεδομένων Πινάκων Κατανάλωσης/Παραγωγής",  style={'textAlign':'center'}),
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
                html.Label("ΔΙΑΘΕΣΙΜΟΙ ΠΙΝΑΚΕΣ ΠΑΡΑΓΩΓΗΣ-ΚΑΤΑΝΑΛΩΣΗΣ",
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
                html.Label("ΜΗΤΡΩΑ ΑΝΤΙΣΤΑΣΗΣ",
                        style={'font-weight': 'bold',
                                'fontSize' : '17px'}),
                dcc.Dropdown(id='availability-radio-resistance',
                            style={"display": "block",
                    "margin-left": "auto",
                    "margin-right": "auto",
                    # "width":"60%"
                    }),
            ], className='four columns'),
        ], className='row',
                 style= {'padding-left' : '50px',
                         'padding-right': '50px'}), # closes the div for first line (matrix and year)
        html.Hr(),
    ],style = {'background-image':image,
                                    'background-size':'cover',
                                    'background-position':'right'}),
    # table here
    html.Hr(),
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
    
    # graphs here
    html.Hr(),
])


@app.callback(
    Output('availability-radio-prods-cons', 'options'),
    Input('availability-radio-prods-cons', 'value'))
def set_products_options(selected_country):
    print(selected_country)
    return [{'label': i, 'value': i} for i in prod_cons_files]



@app.callback(
    Output('availability-radio-resistance', 'options'),
    Input('availability-radio-resistance', 'value'))
def set_products_options(selected_country):
    print(selected_country)
    return [{'label': i, 'value': i} for i in resistance_files]



@app.callback(Output('output-provider', 'children'),
              Input('danger-danger-provider', 'submit_n_clicks'))
def update_output(submit_n_clicks):
    """ documentation: https://dash.plotly.com/dash-core-components/confirmdialogprovider"""
    if not submit_n_clicks:
        return ''
    return """
        Ευχαριστούμε που χρησιμοποιήσατε τις οδηγίες.
    """


@app.callback(
    Output('output-container-slider', 'children'),
    [Input('slider', 'value')]
)
def update_slider(value):
    if value == 0:
        return "Αποτελέσματα για όλους τους μήνες."
    return "Επιλέξατε τον {}o μήνα".format(value)



if __name__ == '__main__':
    app.run_server(debug=True)