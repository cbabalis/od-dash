"""This module processes and shows geospatial data to maps.
It is part of the od-dash application.
Example:

TODO example here
"""

import osrm
import pickle
import polyline
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import network_operations as net_ops
from from_to_pair import FromToPair
import graphs.graph_ops as gops
import networkx as nx
import pdb


css_cols = ['aliceblue','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgrey','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue']


def create_all_min_path_pairs(centroids_csv):
    """Method to get a file with regional units centres and coordinates and to
    return a list of minimum paths between all points with every other point.
    
    Args:
        centroids_csv (str): csv file containing all geo information regarding
        regional units.

    Returns:
        [MinPath]: class with path_between points A and B and their geometry.
        should follow the given od matrix
    """
    # read the csv file to dataframe
    df = load_matrix(centroids_csv)
    # create a list of all pairs
    pairs_list = _create_pairs_from_df(df)
    # for each pair, compute the minimum path
    for pair in pairs_list:
        pair.compute_geometry()
    return pairs_list


def load_matrix(selected_matrix, delim='\t', pre_path='data/geodata_names/'):
    """Method to load a matrix of data as table.

    Args:
        selected_matrix (str): name of the file in disk to be loaded.
        delim (str, optional): delimiter for the file to be read. Defaults to
            '\t'.
        pre_path (str, optional): path leading to file. Defaults to
            'data/geodata_names'.

    Returns:
        dataframe: dataframe containing the file contents.
    """
    matrix_filepath = pre_path + selected_matrix
    sample_df = pd.read_csv(matrix_filepath, delimiter=delim)
    sample_df = sample_df.fillna(0)
    print("full loaded matrix path is ", matrix_filepath)
    return sample_df
 

def _create_pairs_from_df(df):
    # strip df of excess information. hold on to just name and coords
    new_df = df.drop(['ΠΕΡΙΦΕΡΕΙΑ', 'ΚΕΝΤΡΟΕΙΔΕΣ'], axis=1)
    # create a list of tuples containing <name, lon, lat>
    tuples = list(new_df.itertuples(index=False, name=None))
    # iterate tuples and create a list with all vs all
    pairs_list = []
    for curr_node in tuples:
        for next_node in tuples:
            pair = FromToPair('','')
            # assign the tuples to the from and to nodes respectively
            pair.from_node.init_node(*(curr_node))
            pair.to_node.init_node(*(next_node))
            pairs_list.append(pair)
    return pairs_list


def plot_routes_to_map(pairs):
    """Method to plot min paths of pairs to map.

    Args:
        pairs (FromToPair): Minimum path between two nodes and geometry.
        FromToPair class describes the structure of this.
    """
    fig = ''
    graph_filepath = 'data/geodata_names/greece-athens.graphml'
    graph = net_ops.load_graph_from_disk(graph_filepath)
    colorscales = css_cols
    # initialize figure
    fig = go.Figure()
    lat_lines_list = []
    lon_lines_list = []
    lat_nodes_list = []
    lon_nodes_list = []
    skat = 0
    for pair in pairs:
        print(skat)
        skat += 1
        if pair.from_node.name != pair.to_node.name and skat>70:
            has_geometry_set = set_geometry_between_two_points(pair, lat_lines_list, lon_lines_list, skat)
            set_coords_of_two_points(pair, lat_nodes_list, lon_nodes_list, has_geometry_set)
            paint_data_to_figure(fig, lat_nodes_list, lon_nodes_list,
                        lat_lines_list, lon_lines_list, colorscales[-1])
            lat_lines_list = []
            lon_lines_list = []
            lat_nodes_list = []
            lon_nodes_list = []
            if skat> 200:
                break
    #paint_data_to_figure(fig, lat_nodes_list, lon_nodes_list,
    #                     lat_lines_list, lon_lines_list, colorscales)
    return fig


def set_geometry_between_two_points(pair, lat_list, lon_list, skat):
    try:
        route_lines = polyline.decode(pair.geometry)
        print("length of route lines is ", len(route_lines))
    except:
        print("error in pair ", skat)
        return False
    for line in route_lines:
        lat, lon = line
        lat_list.append(lat)
        lon_list.append(lon)
    return True


def set_coords_of_two_points(pair, lat_list, lon_list, has_geometry_set):
    if not has_geometry_set:
        return False
    print("from node is ", pair.from_node.name, " while to node is ", pair.to_node.name)
    lat_list.append(pair.from_node.lat)
    lon_list.append(pair.from_node.lon)
    lat_list.append(pair.to_node.lat)
    lon_list.append(pair.to_node.lon)
    return True


def paint_data_to_figure(fig, lat_nodes_list, lon_nodes_list,
                         lat_lines_list, lon_lines_list, color):
        fig.add_trace(go.Scattermapbox(
            name = "Ροές Φορτίων",
            mode = "lines",
            lon = lon_lines_list,
            lat = lat_lines_list,
            marker = {'size': 10},
            line = dict(width = 4.5, color = color)))
        # adding source marker
        fig.add_trace(go.Scattermapbox(
            name = "Κέντρα Περιφερειακών Ενοτήτων",
            mode = "markers",
            lon = lon_nodes_list,
            lat = lat_nodes_list,
            marker = {'size': 15, 'color':color, 'opacity':0.8},
            ))
        
        
        # getting center for plots:
        lat_center = np.mean(lat_lines_list)
        long_center = np.mean(lon_lines_list)
        # defining the layout using mapbox_style
        fig.update_layout(mapbox_style="open-street-map", #"stamen-terrain",
            mapbox_center_lat = 30, mapbox_center_lon=-80)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                        mapbox = {
                            'center': {'lat': lat_center, 
                            'lon': long_center},
                            'zoom': 8})


def scenario_print_traffic(traffic_fname):
    open_file = open(traffic_fname, 'rb')
    loaded_list = pickle.load(open_file)
    open_file.close()
    fig = plot_routes_to_map(loaded_list)


def simple_scenario(traffic_fname):
    centr_csv = 'centroids.csv'
    mp_pairs = create_all_min_path_pairs(centr_csv)
    open_file = open(traffic_fname, 'wb')
    pickle.dump(mp_pairs, open_file)
    open_file.close()
    print("file has been created")


def print_flows(products_file, centroids_list_file):
    """Method to print in a map the total flows in the network.

    Args:
        products_file (str): csv file that contains all flows towards all.
        centroids_list_file (str): csv file that contains all centroids.
    """
    nodes_list, edges_list, nx_graph = _get_network_as_graph(centroids_list_file)
    _get_flows_between_nodes(products_file, edges_list, nx_graph)


def _get_network_as_graph(centroids_list_file):
    centroids_csv = centroids_list_file #'/home/blaxeep/workspace/od-dash/data/geodata_names/centroids-list.csv'
    centroids_df = pd.read_csv(centroids_csv, sep='\t')
    nodes_list, edges_list, net_graph = gops.read_adjacency_list_from_file(centroids_df)
    return nodes_list, edges_list, net_graph


def _get_flows_between_nodes(products_file, edges_list, net_graph):
    pdf = pd.read_csv(products_file, sep='\t')
    # drop first two columns as they contain redundant data.
    pdf = pdf.iloc[:,2:]
    # get all columns to a list (regional units)
    regional_units = pdf.columns.tolist()
    # iterate over all regional units adding the weight.
    for unit in regional_units:
        values = pdf[unit].tolist()
        _add_weight_to_edges_participating_to_min_path(unit, regional_units, values, edges_list, net_graph)
    pdb.set_trace()


def _add_weight_to_edges_participating_to_min_path(unit, regional_units, values, edges_list, net_graph):
    """method to add weight to edges consisting the minimum path of a flow.
    Algorithm:
    For each <from, to> pair:
        find the min path (dijkstra) from networkx graph
        correspond each edge to edges list
        add the weight to those edges.

    Args:
        unit ([type]): [description]
        values ([type]): [description]
        edges_list ([type]): [description]
        net_graph ([type]): [description]
    """
    # find the min path between the edges
    for reg_unit, weight in zip(regional_units, values):
        try:
            dijkstra_path_nodes = nx.dijkstra_path(net_graph, unit, reg_unit)
            # assign to each edge the weight corresponding to it
            for from_idx in range(len(dijkstra_path_nodes)-1):
                to_idx = from_idx+1
                for edge in edges_list:
                    if edge.are_nodes_in_edge(dijkstra_path_nodes[from_idx], dijkstra_path_nodes[to_idx]):
                        edge.add_to_usage_weight(weight)
        except nx.NetworkXNoPath:
            print("No path between ", unit, " and ", reg_unit)



def main():
    #simple_scenario("/home/blaxeep/Downloads/geolist.pkl")
    #fig = scenario_print_traffic("/home/blaxeep/Downloads/geolist.pkl")
    products_f = '/home/blaxeep/workspace/od-dash/results/mydf.csv'
    centroids_f = '/home/blaxeep/workspace/od-dash/data/geodata_names/perif_centroids.csv'
    print_flows(products_f, centroids_f)
    pdb.set_trace()


if __name__ == '__main__':
    main()