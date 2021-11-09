""" This module executes network operations.

Module to create graph from shapefiles or osm data, to run minimum paths
and to read/write from/to csv files geodata.
"""

import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from ipyleaflet import *
from shapely.geometry import LineString, mapping
import osrm

# debugger. Comment/uncomment depending on case.
import pdb


def get_network_graph(place_name, net_type, custom_filter):
    """ Method to get OSM data and to convert it to network graph.
    
    place_name -- the name of a place.
    net_type -- network type (for instance 'drive')
    custom_filter -- a custom filter for selecting particular parts of the network.
    
    Returns a graph of the selected network.
    """
    ox.config(use_cache=True, log_console=True)
    graph = ox.graph_from_place(place_name, network_type=net_type,
                                custom_filter=custom_filter)
    return graph


def get_nodes_edges(graph):
    """ Returns nodes and edges of a network graph in a tuple."""
    nodes, edges = ox.graph_to_gdfs(graph)
    return (nodes, edges)


def add_new_column_to_dataframe(df, name='traffic'):
    """ Adds a new column to a df and initializes it with 0."""
    df[name] = 0


def get_shortest_path(g, u, v, weight='length'):
    """ Returns a list of nodes that conclude the shortest path between two nodes of a network.
    
    Parameters:
    g -- graph to perform the shortest path
    u -- start node.
    v -- end node.
    weight -- the "cost" param for going from node a to node b. (default: length)
    
    Returns a list of nodes.
    """
    shortest_path = nx.dijkstra_path(g, u, v, weight=weight)
    #shortest_path = nx.astar_path(g, u, v, weight=weight)
    return shortest_path


def get_nodes_pairs(a_list):
    """ Algorithm:
    - create a stop condition (the last element of the input list)
    - iterate the list keeping the current and the next element.
    - return pairs of (current, next) nodes and append them to a new list.
    - when the next node equals the last item of the list (stop condition), then
    - exit the loop and
    - return the list.
    
    :param: list: a list of ints
    :return: list of tuples (containing <u, v> pairs)
    """
    all_pairs = []
    stop_condition = a_list[-1]
    cont = True
    while cont:
        for idx, elem in enumerate(a_list):
            u = elem
            v = a_list[(idx+1) % len(a_list)]
            if v == stop_condition:
                cont = False
            else:
                all_pairs.append((u, v))
    return all_pairs


def update_edge_list(all_pairs_list, edges_list, col_name, cost):
    """ Method to add a value (cost) to each edge in edges_list."""
    for pair in all_pairs_list:
        u, v = pair
        edges_list.loc[(edges_list.u == u) & (edges_list.v == v), col_name] += cost


def split_rows(df, li, ids):
    """ Method to get a list of values and to assign each value to one row."""
    ids_list = ids
    for id in ids_list:
        copied_df = df.copy()
        copied_df.osmid = id
        li.append(copied_df)


def split_osmid_field(loaded_edges):
    """ Method to split osmid field in a dataframe of edges."""
    splitted_rows = []
    print(len(splitted_rows))
    for i in range(len(loaded_edges)):
        df = loaded_edges.iloc[[i]]
        osmid_contents = (df.osmid.to_list()).pop()
        if type(osmid_contents) is not list:
            splitted_rows.append(df)
        else:
            split_rows(df, splitted_rows, osmid_contents)
    return splitted_rows


def get_all_list_combinations(a_list):
    """ Method to get all list combinations by two.
    Returns a new list of tuples with every possible combination.
    """
    combos_list = []
    for u in range(len(a_list)-1):
        for v in range(u+1, len(a_list)):
            combos_list.append((a_list[u], a_list[v]))
    return combos_list


def load_graph_from_disk(src_filepath):
    """ Method to load and return a graph (in graphml) to memory.
    """
    graph = ox.load_graphml(src_filepath)
    return graph


def write_nodes_edges_to_disk(nodes, edges, fname, fpath):
    """ Method to write nodes and edges of a networkx graph to disk.
    
    This method getas as arguments the nodes, the edges, the name of 
    the file and the path to the file and it creates two new files:
    1. 'path_to_file/filename_nodes.csv' and
    2. 'path_to_file/filename_edges.csv'

    Args:
        nodes (dataframe): The nodes of a networkx graph
        edges (dataframe): The edges of a networkx graph
        fname (str): The name of the file.
        fpath (str): The path where the file will be created (it should
        end to '/').
    """
    nodes_fpath = fpath + fname + '_nodes.csv'
    nodes.to_csv(nodes_fpath)
    edges_fpath = fpath + fname + '_edges.csv'
    edges.to_csv(edges_fpath)


def populate_net_nodes_with_sm_nodes(graph, net_nodes, supermarket_nodes):
    """Method to add supermarket id nodes to corresponding network nodes.

    Args:
        graph ([type]): [description]
        net_nodes ([type]): [description]
        supermarket_nodes ([type]): [description]
    """
    # add a new column to network nodes
    add_new_column_to_dataframe(net_nodes, name='supermarket_id')
    # assign supermarket id node to corresponding node in network
    update_net_nodes_with_sm_id_nodes(net_nodes, supermarket_nodes, graph)
    # return network nodes
    return net_nodes


def update_net_nodes_with_sm_id_nodes(net_nodes, supermarket_nodes, graph):
    """[summary]

    Args:
        net_nodes (dataframe): [description]
        supermarket_nodes (dataframe): [description]
    """
    for row_id, sm in supermarket_nodes.iterrows():
        id = 0
        # for each supermarket get location from geometry
        geometry = sm.geometry
        loc = get_supermarket_location(geometry)
        # find nearest network node to this location
        nearest_net_node = ox.get_nearest_node(graph, loc)
        net_nodes.loc[net_nodes['osmid'] == nearest_net_node, ['supermarket_id']] = sm.id
    return net_nodes


def get_random_point_from_polygon_geometry(geometry):
    """Method to get a random point (x, y) from a polygon.

    Args:
        geometry (Polygon): A row of a dataframe

    Returns:
        tuple: (y, x) geometry in a tuple
    """
    point = 0
    coord_sequence = geometry.exterior.coords
    for cs in coord_sequence:
        point = cs
    return point


def get_supermarket_location(geometry):
    if geometry.geom_type == 'Polygon':
        point = get_random_point_from_polygon_geometry(geometry)
        return point
    elif geometry.geom_type == 'Point':
        point = (geometry.y, geometry.x)
        return point
    return -1


def create_adj_matrix_of_supermarkets(sm_nodes, graph):
    """method to create the adjacency matrix of nodes that represent
    the supermarkets.

    Args:
        net_nodes ([type]): [description]
        graph ([type]): [description]

    Returns:
        [type]: [description]
    """
    # get ids of all supermarkets and create pairs
    adj_matrix = create_empty_adj_matrix(sm_nodes)
    # run a dijkstra between all pairs
    distances = create_distance_matrix(adj_matrix, graph)
    # return the table
    return distances


def create_empty_adj_matrix(sm_nodes, col='osmid'):
    """Method to create a len(col) x len(col) symmetric matrix and to
    initialize it.

    Args:
        sm_nodes ([type]): [description]
        col (str, optional): [description]. Defaults to 'osmid'.

    Returns:
        [type]: [description]
    """
    df = pd.crosstab(sm_nodes[col], sm_nodes[col])
    idx = df.columns.union(df.index)
    df = df.reindex(index=idx, columns=idx, fill_value=99999)
    return df


def create_distance_matrix(df, graph):
    # get all nodes ids to a list.
    all_nodes = df.columns.to_list()
    # check if all nodes of interest in network have path that connects them
    all_nodes = fast_check_network_integrity(all_nodes, graph, threshold=10)
    # for each node acquire a list with the distance between it and the other nodes
    distances = []
    for node in all_nodes:
        distance_list = compute_distance_from_other_nodes(node, all_nodes, graph)
        distances.append(distance_list)
    return distances


def compute_distance_from_other_nodes(node, node_list, graph):
    dist_list = []
    no_pairs_dict = {}
    no_pairs_list = []
    no_conns_found = 0
    for n in node_list:
        try:
            dist = nx.shortest_path_length(graph, node, n)
        except nx.exception.NetworkXNoPath:
            #print("no path between %s and %s", node, n)
            dist = 999999
            no_conns_found += 1
            no_pairs_list.append(n)
        dist_list.append(dist)
    if no_conns_found:
        print("connections not found: ", no_conns_found, " out of ", len(node_list))
        no_pairs_dict[node] = no_pairs_list
        print("connections not found between", no_pairs_dict)
    return dist_list


def fast_check_network_integrity(node_list, graph, threshold=17):
    """Method to check in a fast, sloppy way if there is path from any point of the network to any other.
    Return a list of the nodes that exist paths connecting them

    Args:
        node_list (list): List of node ids.
        graph (graphml): graph of the network
    """
    connected_list = []
    no_conns_found = 0
    # iterate each node to each other
    # if there is no connection between two nodes,
    # if not connected nodes are less than threshold, then remove the not connected nodes
    # from the list.
    lonely_nodes = []
    for start_node in node_list:
        no_connected_list = []
        for end_node in node_list:
            try:
                dist = nx.shortest_path_length(graph, start_node, end_node)
            except nx.exception.NetworkXNoPath:
                no_conns_found += 1
                no_connected_list.append(end_node)
        if len(no_connected_list) > 0:
            if len(no_connected_list) < threshold:
                node_list = [x for x in node_list if x not in no_connected_list]
            else:
                lonely_nodes.append(start_node)
    _remove_nodes_from_list(lonely_nodes, node_list)
    return node_list
    

def _remove_nodes_from_list(list_to_remove, nodes_list):
    """[summary]

    Args:
        list_to_remove ([type]): [description]
        nodes_list ([type]): [description]
    """
    for elem in list_to_remove:
        if elem in nodes_list:
            nodes_list.remove(elem)
        else:
            print("elem not in list: ", elem)


def compute_distance_matrix(csv_nodes_path, titles='', annotations='distance'):
    """ Method to compute and return distance matrix given a set of nodes of interest

    Args:
        csv_nodes_path (str]): Path to a csv containing info about nodes.
        titles (str, optional): Titles of the od matrix to be produced. Defaults to ''.
        annotations (str, optional): distance or time. Defaults to 'distance'.
    """
    supermarkets = pd.read_csv(csv_nodes_path, delimiter='\t')
    coords = [','.join(str(x) for x in y) for y in map(tuple, supermarkets[['latitude', 'longitude']].values)]
    numerical_coords = []
    for elem in coords:
        long_lat_elems = elem.split(',')
        numerical_coords.append((float(long_lat_elems[1]), float(long_lat_elems[0])))
    # create titles
    if not titles or len(titles) != len(numerical_coords):
        titles = [i for i in range(len(numerical_coords))]
    # compute distance matrix and return it
    dist_od = osrm.table(numerical_coords, ids_origin=titles, output='pandas', annotations=annotations)
    return dist_od


def compare_elems(a_list):
    counter = 0
    similarities = {}
    for this in a_list:
        for that in a_list:
            if this == that:
                counter += 1
        if counter > 2:
            similarities[this] = counter
        counter = 0
    return similarities


def add_u_v_coords_to_edges(nodes, edges):
    """ #TODO maybe this and the following methods need to be removed. check

    Args:
        nodes ([type]): [description]
        edges ([type]): [description]
    """
    # populate edges with coordinate fields
    coord_fields = ['u_x', 'u_y', 'v_x', 'v_y']
    _populate_edges_with_new_fields(edges, coord_fields)
    # for each edge get u, v nodes
    pdb.set_trace() # TODO too much code needs to be written!
    for edge in edges.iterrows():
        # for each node get its coordinates and add them to the current edge
        u = edge['u']
        populate_edge(u, edge, x='u_x', y='u_y')
        v = edge['v']
        populate_edge(v, edge, x='v_x', y='v_y')


def _populate_edges_with_new_fields(edges, coord_fields):
    for field in coord_fields:
        if field not in edges.columns:
            edges[field] = None


def populate_edge(node, edge, x, y):
    n_x = node.X
    n_y = node.Y
    edge[x] = n_x
    edge[y] = n_y
