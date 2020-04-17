import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import json
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from country_list import countries_for_language
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
# DATA_FILE = 'labeled_npi.csv'

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
cluster_df = pd.read_pickle('final_df.pkl')

def get_breaks(row, col, word_limit=45, break_char='<br>', colon=True):
    col_list = ['tokens', 'author_list', 'doi', 'countries']
    if row[col] == row[col]:
        data = ''
        if col in col_list:
            if col != 'tokens':
                data = f'**{col.capitalize()}:** '
            words = row[col]
        elif colon:
            if break_char == '<br>':
                data = f'{col.capitalize()}: '
            else:
                data = f'**{col.capitalize()}:** '
            words = row[col].split(' ')
        else:
            words = row[col].replace('<br>', '').split(' ')
            words[0] = f'**{words[0]}**'
        total_chars = 0

        # add break every length characters
        for i in range(len(words)):
            total_chars += len(words[i])
            if total_chars > word_limit:
                data += f'{break_char}{words[i]}'
                total_chars = 0
            else:
                if col in col_list and data:
                    data += f' {words[i]},'
                else:
                    data += f' {words[i]}'
        return data.strip().rstrip(',')
    return row[col]

def get_country_df(df):
    country = []
    count = []
    for k in dict(countries_for_language('en')).values():
        len_country = len(df[df['countries'].map(set([k.lower()]).issubset)])
        country.append(k.lower())
        count.append(len_country)
    return pd.DataFrame({'country': country, 'count': count})

def get_wordcloud(df):
    text = ''
    for body_text in df['title']:
        text = text + body_text
    tokens = word_tokenize(text)
    remove_sw = [word.lower() for word in tokens if word not in stopwords.words('english')]
    remove_punct = [word for word in remove_sw if not word in [',', '(', ')', '"', ':', '``', '.', '?', '<', '>', 'br', 'title']]
    keywords = Counter(remove_punct).most_common()[0:5]
    return ', '.join([keyword[0] for keyword in keywords])

class Cluster_Plot:
    def __init__(self, df, text_type, clust_nums):
        self.cluster_id = 'all'
        self.dimension = '3d'
        self.styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }
        self.search = ''
        self.df = df
        self.set_text(text_type)
        self.clust_nums = clust_nums
        self.cluster_id_list = [{'label': i, 'value': i} for i in list(range(self.clust_nums))]
        self.cluster_id_list.append({'label': 'all', 'value': 'all'})
        self.create_cluster_df()
        self.app = dash.Dash(__name__,
                        external_stylesheets=[dbc.themes.BOOTSTRAP, "https://codepen.io/chriddyp/pen/bWLwgP.css"])
        self.set_app_layout()                
        self.server = self.app.server
        if self.app is not None and hasattr(self, 'callbacks'):
            self.callbacks(self.app)


    def run_process(self):
        self.app.config.suppress_callback_exceptions = True
        self.app.run_server()

    # def set_app(self):
    #     self.app = app


    def set_app_layout(self):
        self.app.layout = html.Div(children=[
            html.Div(className='row', style={'background-color': '#142a57'}, children=[
                html.Div(className='container', style={'max-width': 'unset'}, children=[
                    html.Div(className='row', children=[
                        html.Div(className='nine columns', children=[
                            html.H1('NPI Cluster Analysis', style={ 'color': 'white', 'padding-top': '1%' })
                        ])
                    ])
                ])
            ]),
            html.Div(className='container', style={'max-width': 'unset'}, children=[
                html.Div(className='row', children=[
                    html.Div(className='three columns', style={'padding-top': '5%'}, children=[
                        dcc.Tabs(id="tabs", value='tab-1', children=[
                            dcc.Tab(label='Edit Clusters', value='tab-1'),
                            dcc.Tab(label='Filter Articles', value='tab-2'),
                        ])
                    ]),
                    html.Div(className='six columns', children=[
                        html.Div(className='four columns', children=[
                            dcc.Graph(id="geo-graph", style={'height': '60px'})
                        ]),
                        html.Div(className='eight columns', children=[
                            html.Button(id='geo-button', style={'margin-left': '40%', 'margin-top': '20%'}, children='Reset Map')
                        ])
                    ])
                ]),
                html.Div(className='row', style={'padding-top': '5%'}, children=[
                    html.Div([
                        dcc.Markdown("""
                            **Selected Article.**

                            Click on values in the plot to select article.
                        """),
                        dcc.Markdown(id='hover-data', style=self.styles['pre'])
                    ], className='eight columns'),
                    html.Div(className='four columns', style={"float": "right", 'left': '60%', 'top': '20px'}, children=[
                        dcc.Graph(id="graph", style={'width': '75%', 'height': '350px'}),
                        html.H5('Current Cluser keywords'),
                        html.Div(id='cluster_keywords'),
                        html.Div(id='tabs-content')
                    ])
                ]),
                html.Div(className='row', style={'margin-top': '10%'}, children=[
                    html.Div(className='five columns', style={'display': 'none'}, children=[
                         html.Div(id='dummy_dimension', style={'display':'none'}),
                         html.Div(id='dummy_cluster_num', style={'display':'none'}),
                         html.Div(id='dummy_text', style={'display':'none'}),
                         html.Div(id='dummy_cluster_id', style={'display':'none'}),
                         html.Div(id='dummy_search_string', style={'display':'none'}),
                         html.Div(id='dummy_geo_graph', style={'display':'none'}),
                         dcc.RadioItems(
                                id='dimension',
                                options=[{'label': ' 2d', 'value': '2d'},
                                         {'label': ' 3d', 'value': '3d'}],
                                value= self.dimension,
                                style={'display': 'none'}
                            ),
                            dcc.RadioItems(
                                id='abstract_or_body',
                                options=[{'label': ' Abstract', 'value': 'abstract'},
                                         {'label': ' Body', 'value': 'body_text'}],
                                value=self.text_type,
                                style={'display': 'none'}
                            ),
                            dcc.Dropdown(
                                id='cluster_num',
                                options=[{'label': i+1, 'value': i+1} for i in list(range(20))],
                                value=self.clust_nums,
                                style={'display': 'none'}
                            ),
                            dcc.Dropdown(
                                id='cluster_id',
                                options=self.cluster_id_list,
                                value=self.cluster_id,
                                style={'display': 'none'}
                            ),
                        dcc.Input(id='search',
                                value='',
                                type='text',
                                style={'display': 'none'}
                            )
                    ])
                ])
            ])
        ])

    def create_cluster_df(self):
        new_df = self.df.copy()
        kmeans = KMeans(n_clusters = self.clust_nums, random_state=1)
        new_df[self.col_cluster_id] = list(kmeans.fit_predict(np.stack(new_df[self.col].to_numpy())))
        self.cluster_df = new_df.reset_index(drop=True)
        self.cluster_df['title'] = self.cluster_df.apply(get_breaks, args=('title',), axis=1)
        self.cluster_df[['x', 'y', 'z']] = pd.DataFrame(self.cluster_df[self.col].values.tolist(),
                                                        index = self.cluster_df.index)

    def set_text(self, text_type):
        self.text_type = text_type
        self.col = f'{self.text_type}_tfidf_pca_scaled'
        self.col_cluster_id = f'{self.text_type}_tfidf_pca_scaled_clusterID'

    def callbacks(self, app):
        @app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
        def render_content(tab):
            articles_content = html.Div(className='filter-content', style={'background-color': 'white'}, children=[
                            html.H4('Filter Articles'),
                            html.Label('Search keywords'),
                            dcc.Input(id='search',
                                value='',
                                type='text'
                            ),
                            html.Button(id='submitBtn', children='submit')
                        ])
            cluster_content = html.Div(id='cluster-content', style={'width': '100%', 'background-color': 'white'}, children=[
                            html.H4('Cluster Control Panel'),
                            html.Label('Select to show 2d or 3d visualization'),
                            dcc.RadioItems(
                                id='dimension',
                                options=[{'label': ' 2d', 'value': '2d'},
                                         {'label': ' 3d', 'value': '3d'}],
                                value='3d'
                            ),
                            html.Label('Select to cluster on the article abstract or body'),
                            dcc.RadioItems(
                                id='abstract_or_body',
                                options=[{'label': ' Abstract', 'value': 'abstract'},
                                         {'label': ' Body', 'value': 'body_text'}],
                                value=self.text_type
                            ),
                            html.Label('Select number of Clusters'),
                            dcc.Dropdown(
                                id='cluster_num',
                                options=[{'label': i+1, 'value': i+1} for i in list(range(20))],
                                value=self.clust_nums
                            ),
                            html.Label('Select cluster to view'),
                            dcc.Dropdown(
                                id='cluster_id',
                                options=self.cluster_id_list,
                                value='all'
                            )
                        ])
            if tab == 'tab-1':
                return cluster_content
            elif tab == 'tab-2':
                return articles_content


        @app.callback([Output('cluster_id', 'options'),
                       Output('dummy_cluster_num', 'children')],
                      [Input('cluster_num', 'value')])
        def set_cluster_options(cluster_num):
            self.clust_nums = cluster_num
            self.cluster_id_list = [{'label': i, 'value': i} for i in list(range(self.clust_nums))]
            self.cluster_id_list.append({'label': 'all', 'value': 'all'})
            return self.cluster_id_list, ''

        @app.callback(Output('dummy_dimension', 'children'),
                      [Input('dimension', 'value')])
        def set_dimension(dimension):
            self.dimension = dimension

        @app.callback(Output('dummy_cluster_id', 'children'),
                      [Input('cluster_id', 'value')])
        def set_cluster_id(cluster_id):
            self.cluster_id = cluster_id

        @app.callback(Output('dummy_text', 'children'),
                      [Input('abstract_or_body', 'value')])
        def set_abstract_or_body(abstract_or_body):
            self.set_text(abstract_or_body)

        @app.callback(Output('dummy_search_string', 'children'),
                      [Input('submitBtn', 'n_clicks')],
                      [State('search', 'value')])
        def set_search(clicks, search):
            self.search = search

        @app.callback(Output('dummy_geo_graph', 'children'),
                      [Input('geo-graph', 'clickData'),
                       Input('geo-button', 'n_clicks')])
        def set_geo_graph(geo_click_data, clicks):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'geo-button' not in changed_id:
                self.geo_click_data = geo_click_data
            else:
                self.geo_click_data = ''

        @app.callback([Output('graph', 'figure'), Output('geo-graph', 'figure'), Output('cluster_keywords', 'children')],
                      [Input('dummy_dimension', 'children'),
                       Input('dummy_cluster_num', 'children'),
                       Input('dummy_cluster_id', 'children'),
                       Input('dummy_text', 'children'),
                       Input('dummy_search_string', 'children'),
                       Input('dummy_geo_graph', 'children')
                       ])
        def update_clusters(dummy1, dummy2, dummy3, dummy4, dummy5, dummy6):
            self.create_cluster_df()
            df = self.cluster_df.copy()
            show_scale = True
            cluster_keywords = None
            if self.cluster_id != 'all':
                show_scale = False
                df = df[df[self.col_cluster_id] == self.cluster_id]
                cluster_keywords = get_wordcloud(df)
            if self.geo_click_data:
                country = self.geo_click_data['points'][0]['location']
                if len(df[df['countries'].map(set([country]).issubset)]):
                    df = df[df['countries'].map(set([country]).issubset)]
            if self.search:
                if len(df[df['tokens'].map(set([self.search]).issubset)]):
                    df = df[df['tokens'].map(set([self.search]).issubset)]
            country_df = get_country_df(df)
            if self.dimension == '2d':
                fig = px.scatter(df, x='x', y='y',
                                 color=self.col_cluster_id,
                                 hover_name='title',
                                 hover_data=['paper_id', 'doi'])
                fig.update_layout(title = '2D cluster of research papers',
                                  xaxis = dict(dtick=1, range=[-5,5], scaleratio = 1),
                                  yaxis = dict(dtick=1, range=[-5,5], scaleratio = 1),
                                  hoverlabel=dict(
                                    bgcolor='white',
                                    font_size=8,
                                    font_family='Rockwell'
                                  ),
                                  coloraxis=dict(
                                    colorbar=dict(title='Cluster ID'),
                                    showscale=show_scale
                                  ))
            elif self.dimension == '3d':
                fig = px.scatter_3d(df, x='x', y='y', z='z',
                                    color=self.col_cluster_id,
                                    hover_name='title',
                                    hover_data=['paper_id', 'doi'])
                fig.update_layout(title = '3D cluster of research papers',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  scene = dict(
                                    xaxis = dict(dtick=1, range=[-5,5],),
                                    yaxis = dict(dtick=1, range=[-5,5],),
                                    zaxis = dict(dtick=1, range=[-5,5],),),
                                  hoverlabel=dict(
                                    bgcolor='white',
                                    font_size=8,
                                    font_family='Rockwell'
                                  ),
                                  coloraxis=dict(
                                    colorbar=dict(title='Cluster ID'),
                                    showscale=show_scale
                                  ))
            fig2 = px.scatter_geo(country_df,
                                  locationmode='country names',
                                  locations='country',
                                  hover_name='country',
                                  size='count',
                                  projection='natural earth')
            fig2.update_layout(autosize=False,
                               width=500,
                               height=250,
                               paper_bgcolor='rgba(0,0,0,0)'
                              )
            self.search = ''
            return fig, fig2, cluster_keywords

        @app.callback(Output("hover-data", "children"),
                      [Input("graph", "clickData")])
        def display_click_data(clickData):
            string = None
            if clickData:
                click_paper_id = clickData['points'][0]['customdata'][0]
                click_index = self.cluster_df[self.cluster_df['paper_id'] == click_paper_id].index[0]
                token_string = ''
                country_string = ''
                if self.cluster_df.iloc[click_index]["tokens"]:
                    token_string = get_breaks(self.cluster_df.iloc[click_index],
                                              'tokens',
                                              word_limit=100,
                                              break_char='\n')
                    token_string = f'**Keywords**: {token_string}'
                if self.cluster_df.iloc[click_index]["countries"]:
                    country_string = get_breaks(self.cluster_df.iloc[click_index],
                                                'countries',
                                                word_limit=100,
                                                break_char='\n')

                string = get_breaks(self.cluster_df.iloc[click_index],
                                    'title',
                                    word_limit=100,
                                    break_char='\n',
                                    colon=False)
                item_list = ['abstract', 'body_text', 'author_list', 'paper_id', 'doi']
                for i in item_list:
                    formatted_data = get_breaks(self.cluster_df.iloc[click_index],
                                                i,
                                                word_limit=100,
                                                break_char='\n')
                    string += f'\n\n{formatted_data}'
                if token_string:
                    string = f'{token_string}\n\n{string}'
                if country_string:
                    string = f'{country_string}\n\n{string}'
                return string
            return string

app = Cluster_Plot(cluster_df, 'abstract', 10)
server = app.server
if __name__ == '__main__':
    print('app starting..')
    app.run_process()
