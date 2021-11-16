"""
Module to implement graph operations as well as read/write operations.
"""

import pandas as pd

import network_graph as net_graph
import pdb


def read_adjacency_list_from_file(df):
    """Reads an adjacency list from a file.
    Returns a pandas dataframe.

    Args:
        df (Dataframe): Representation of regional units, centroids, their
        coordinates as well as their neighbors.
    """
    # get basic information (centroids, coords) from dataframe
    # https://www.datasciencelearner.com/drop-unnamed-column-pandas/
    centroids_df = df.loc[:, ~df.columns.str.match("Unnamed")]
    # get neighbors from the dataframe
    neighbors_df = df[df.columns[pd.Series(df.columns).str.startswith("Unnamed")]]
    # and convert it to list of lists
    neighbors_list = neighbors_df.values.tolist()
    # finally get rid of nan values
    cleaned_neighbors_list = []
    for neighbors in neighbors_list:
        cleaned = [neighbor for neighbor in neighbors if str(neighbor) != 'nan']
        cleaned_neighbors_list.append(cleaned)
    # create adjacency list
    create_graph(centroids_df, cleaned_neighbors_list)


def create_graph(centroids_df, neighbors_list):
    """Method to create a graph of vertices, edges representing the network.
    Centroids_df converts to nodes and neighbors_list are the neighbors of
    each centroid.

    Args:
        centroids_df ([type]): [description]
        neighbors_list ([type]): [description]
    """
    # create a list of tuples containing <name, lon, lat>
    new_df = centroids_df.drop(['ΠΕΡΙΦΕΡΕΙΑΚΕΣ ΕΝΟΤΗΤΕΣ', 'ΠΕΡΙΦΕΡΕΙΑ'], axis=1)
    tuples = list(new_df.itertuples(index=False, name=None))
    # iterate tuples and create a list with all vs all
    nodes_list = []
    populate_nodes_list(nodes_list, tuples)
    pdb.set_trace()
    edges_list = []
    populate_edges_list(edges_list, nodes_list, neighbors_list) #TODO here


def  populate_nodes_list(nodes_list, tuples):
    for t in tuples:
        name, lon, lat = t
        vertex = net_graph.Vertex(name=name, lat=lat, lon=lon)
        nodes_list.append(vertex)
    


def main():
    # read the csv file as dataframe
    centroids_csv = '/home/blaxeep/workspace/od-dash/data/geodata_names/centroids-list.csv'
    centroids_df = pd.read_csv(centroids_csv, sep='\t')
    read_adjacency_list_from_file(centroids_df)
    pdb.set_trace()
    # convert dataframe contents to nodes and edges
    #return the result


if __name__ == '__main__':
    main()