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
import copy
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
            if skat> 100:
                break
    #paint_data_to_figure(fig, lat_nodes_list, lon_nodes_list,
    #                     lat_lines_list, lon_lines_list, colorscales)
    return fig


def set_geometry_between_two_points(pair, lat_list, lon_list, skat):
    try:
        route_lines = polyline.decode(pair.geometry)
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
    #print("from node is ", pair.from_node.name, " while to node is ", pair.to_node.name)
    lat_list.append(pair.from_node.lat)
    lon_list.append(pair.from_node.lon)
    lat_list.append(pair.to_node.lat)
    lon_list.append(pair.to_node.lon)
    return True


def paint_data_to_figure(fig, lat_nodes_list, lon_nodes_list,
                         lat_lines_list, lon_lines_list, color, weight=4.5,
                         node_title="Κέντρα Περιφερειακών Ενοτήτων", edge_title="Ροές Φορτίων"):
        fig.add_trace(go.Scattermapbox(
            name = edge_title,
            mode = "lines",
            lon = lon_lines_list,
            lat = lat_lines_list,
            marker = {'size': 10},
            line = dict(width = weight, color = color)))
        # adding source marker
        fig.add_trace(go.Scattermapbox(
            name = node_title,
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


def print_flows(products_file, nodes_list, edges_list, nx_graph):
    """Method to print in a map the total flows in the network.

    Args:
        products_file (str): csv file that contains all flows towards all.
        centroids_list_file (str): csv file that contains all centroids.
    """
    _get_flows_between_nodes(products_file, edges_list, nodes_list, nx_graph)
    # prepare each edge geometry and coordinates for printing
    colorscales = css_cols
    old_range, min_val = _compute_old_range(edges_list)
    # sort edges list by usage weight (in order for the heavier edges to be last)
    edges_list.sort(key=lambda x: x.usage_weight, reverse=False)
    fig = go.Figure()
    for edge in edges_list:
        lat_nodes_list = [edge.from_node.lat, edge.to_node.lat]
        lon_nodes_list = [edge.from_node.lon, edge.to_node.lon]
        lat_lines_list, lon_lines_list = edge.get_geometry_lats_lons_lists()
        scaled_weight = _get_scaled_weight(edge, old_range, min_val)
        scaled_color = _set_scaled_color(scaled_weight)
        route_title_name, route_name_weight = _get_route_details(edge)
        if edge.print_enabled and edge.usage_weight > 0:
            paint_data_to_figure(fig, lat_nodes_list, lon_nodes_list,
                            lat_lines_list, lon_lines_list, scaled_color, scaled_weight,
                            route_title_name, route_name_weight)
    return fig, edges_list, nodes_list


def get_network_as_graph(centroids_list_file):
    centroids_csv = centroids_list_file #'/home/blaxeep/workspace/od-dash/data/geodata_names/centroids-list.csv'
    centroids_df = pd.read_csv(centroids_csv, sep='\t')
    nodes_list, edges_list, net_graph = gops.read_adjacency_list_from_file(centroids_df)
    return nodes_list, edges_list, net_graph


def _get_flows_between_nodes(products_file, edges_list, nodes_list, net_graph):
    pdf = pd.read_csv(products_file, sep='\t')
    # drop first two columns as they contain redundant data.
    pdf = pdf.iloc[:,2:]
    # get all columns to a list (regional units)
    regional_units = pdf.columns.tolist()
    # iterate over all regional units adding the weight.
    for unit in regional_units:
        values = pdf[unit].tolist()
        _add_weight_to_edges_participating_to_min_path(unit, regional_units, values, edges_list, nodes_list, net_graph)


def _add_weight_to_edges_participating_to_min_path(unit, regional_units, values, edges_list, nodes_list, net_graph):
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
            _assign_weight_type_to_node(nodes_list, copy.deepcopy(dijkstra_path_nodes), weight)
            # assign to each edge the weight corresponding to it
            for from_idx in range(len(dijkstra_path_nodes)-1):
                to_idx = from_idx+1
                for edge in edges_list:
                    if edge.are_both_nodes_in_edge(dijkstra_path_nodes[from_idx], dijkstra_path_nodes[to_idx]):
                    #if edge.is_node_in_edge(dijkstra_path_nodes[from_idx]) or\
                    #    edge.is_node_in_edge(dijkstra_path_nodes[to_idx]):
                        edge.add_to_usage_weight(weight)
        except nx.NetworkXNoPath:
            print("No path between ", unit, " and ", reg_unit)



def _compute_old_range(edges_list):
    min_val = 1000000
    max_val = 0
    # find minimum and maximum weights from total edges
    for edge in edges_list:
        if min_val > edge.usage_weight:
            min_val = edge.usage_weight
        if max_val < edge.usage_weight:
            max_val = edge.usage_weight
    # compute old_range and return it
    old_range = max_val - min_val
    print("max range is ", max_val, " and min val is ", min_val)
    return old_range, min_val


def _get_scaled_weight(edge, old_range, min_val, new_range=12):
    """scaling takes place according to this:
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    
    https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio

    Args:
        edge ([type]): [description]
        old_range ([type]): [description]
        min_range ([type]): [description]
    """
    if old_range == 0:
        print("no range, it is 0. cannot be.")
        return -1
    new_min = 3
    old_weight = edge.usage_weight
    weight = ((old_weight - min_val) * new_range) / old_range +new_min
    return weight


def _set_scaled_color(weight, max_threshold=12):
    if weight < max_threshold/4 + 3:
        return 'rgb(0, 255, 0)'
    elif weight < 2*max_threshold/4 + 3:
        return 'rgb(255, 255, 0)'
    elif weight < 3*max_threshold/4 + 3:
        return 'rgb(255, 165, 0)'
    else:
        return 'rgb(255, 0, 0)'


def _get_route_details(edge):
    """ method to get route name and weight."""
    route_name = "Κέντρο Περιφερειακής Ενότητας"
    #route_flow = "Κίνηση: " + str(round(edge.usage_weight, 2))
    if np.isnan(edge.usage_weight):
        edge.usage_weight = 0
    route_flow = str(edge.edge_name) + ":" + str(int(edge.usage_weight)) + " κιλά"
    return route_name, route_flow


def _assign_weight_type_to_node(nodes_list, dijkstra_path_list, weight):
    """
    Method to assign weight to nodes depending on the type of the node.
    If it is a "from" node (meaning weight is starting from there),
    a "to" node (meaning weight is ending there) or
    a "passing" node (meaning this is a node where the weight is transferred through).

    Networkx dijkstra method returns a list of nodes for a certain path, including
    from and to nodes. For instance, in a route A -> B -> C -> D the dijkstra method
    return a list of [A, B, C, D].

    nodes_list is a list containing all available nodes.
    dijkstra path is a path of nodes in a certain route.
    weight is the weight to be added to each node.
    """
    if len(dijkstra_path_list) < 2:
        single_node = dijkstra_path_list.pop()
        #print("number of elements of list dijksta is ", len(dijkstra_path_list), " and it is ", str(single_node))
        #_assign_weight_to_node(single_node, nodes_list, weight, weight_type='from')
        #_assign_weight_to_node(single_node, nodes_list, weight, weight_type='to')
    else:
        _assign_from_to_nodes(dijkstra_path_list, nodes_list, weight)
        if len(dijkstra_path_list) > 0:
            for dijkstra_node in dijkstra_path_list:
                # assign the weight to the right value in the node and continue.
                _assign_weight_to_node(dijkstra_node, nodes_list, weight, weight_type='passing')


def _assign_from_to_nodes(dijkstra_path_list, nodes_list, weight):
        last_node = dijkstra_path_list.pop()
        _assign_weight_to_node(last_node, nodes_list, weight, weight_type='to')
        first_node = dijkstra_path_list.pop(0)
        _assign_weight_to_node(first_node, nodes_list, weight, weight_type='from')


def _assign_weight_to_node(dijkstra_node, nodes_list, weight, weight_type):
    for node in nodes_list:
        if node.name == dijkstra_node:
            if weight_type == 'from':
                node.from_weight += weight
                break
            elif weight_type == 'to':
                node.to_weight += weight
                break
            else:
                node.passing_weight += weight
                break


def main():
    #simple_scenario("/home/blaxeep/Downloads/geolist.pkl")
    fig = scenario_print_traffic("/home/blaxeep/Downloads/geolist.pkl")
    products_f = '/home/blaxeep/workspace/od-dash/results/output-1.csv' #mydf.csv'
    centroids_f = '/home/blaxeep/workspace/od-dash/data/geodata_names/perif_centroids_abal.csv'
    nodes_list, edges_list, nx_graph = get_network_as_graph(centroids_f)
    fig, edges_list, nodes_list = print_flows(products_f, nodes_list, edges_list, nx_graph)
    pdb.set_trace()


if __name__ == '__main__':
    main()