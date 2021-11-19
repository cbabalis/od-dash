"""
Module to implement graph operations as well as read/write operations.
"""

import osrm
import networkx as nx
import pandas as pd
import graphs.network_graph as net_graph
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
    nodes_list, edges_list, net_graph = create_graph(centroids_df, cleaned_neighbors_list)
    return nodes_list, edges_list, net_graph


def create_graph(centroids_df, neighbors_list):
    """Method to create a graph of vertices, edges representing the network.
    Centroids_df converts to nodes and neighbors_list are the neighbors of
    each centroid.

    Args:
        centroids_df ([type]): [description]
        neighbors_list ([type]): [description]
    """
    # create a list of tuples containing <name, lon, lat>
    new_df = centroids_df.drop(['ΚΕΝΤΡΟΕΙΔΕΣ', 'ΠΕΡΙΦΕΡΕΙΑ'], axis=1)
    tuples = list(new_df.itertuples(index=False, name=None))
    # create all nodes and add them to a list
    nodes_list = []
    populate_nodes_list(nodes_list, tuples)
    edges_list = []
    # create neighbors to the nodes
    assign_neighbors_to_nodes(nodes_list, neighbors_list, tuples)
    # create edges list
    populate_edges_list(edges_list, nodes_list, neighbors_list, tuples)
    # have them all inside networkx
    net_graph = add_networkx_graph(nodes_list, edges_list)
    return (nodes_list, edges_list, net_graph)
    


def  populate_nodes_list(nodes_list, tuples):
    for t in tuples:
        name, lon, lat = t
        vertex = net_graph.Vertex(name=name, lat=lat, lon=lon)
        nodes_list.append(vertex)


def assign_neighbors_to_nodes(nodes_list, neighbors_list, tuples):
    """Method to assign neighbors to nodes.
    
    Nodes list has all node instances of the network.
    Neighbors list is a list with the names of the neighbors.
    Element of the first list points to the corresponding element of the second
    list, etc.

    Args:
        nodes_list ([type]): [description]
        neighbors_list ([type]): [description]
    """
    # iterate both lists
    for node, neighbors in zip(nodes_list, neighbors_list):
        # for each node, search the node to the first list
        for neighbor in neighbors:
            # make a minimum path between the two and
            # save it to the node's list.
            neighbor_name_coords = [item for item in tuples if item[0] == neighbor][0]
            assign_distance_between_neighbors(node, neighbor_name_coords)


def assign_distance_between_neighbors(node, neighbor_name_coords):
    # get node coords
    node_point = osrm.Point(latitude=node.lat, longitude=node.lon)
    # get neighbor coords
    neighbor_name, neighbor_lon, neighbor_lat = neighbor_name_coords
    neighbor_point = osrm.Point(latitude=neighbor_lat, longitude=neighbor_lon)
    # find distance between the two
    full_route = osrm.simple_route(node_point, neighbor_point)
    dist = full_route['routes'][0]['distance']
    # populate node with the neighbor and the distance between the two.
    node.set_new_neighbor_dist(neighbor_name, dist)
    # populate node with the neighbor and the duration between the two.
    dur = full_route['routes'][0]['duration']
    node.set_new_neighbor_dur(neighbor_name, dur)


def populate_edges_list(edges_list, nodes_list, neighbors_list, tuples):
    # for each node create an edge with its neighbor.
    for node, neighbors in zip(nodes_list, neighbors_list):
        # for each node, search the node to the first list
        for neighbor in neighbors:
            neighbor_name_coords = [item for item in tuples if item[0] == neighbor][0]
            # populate edge with from-to nodes, name of the edge and geometry
            from_node = node
            neighbor_name = neighbor_name_coords[0]
            to_node = [a_node for a_node in nodes_list if a_node.name==neighbor_name][0]
            # check if edge exists backwards (from-to is to-from) and if not, continue
            if _edge_exists(edges_list, from_node, to_node):
                #print("edge exists between", str(from_node.name), " and ", str(to_node.name))
                continue
            edge = net_graph.Edge(from_node, to_node)
            # save geometry to edge.
            edge.compute_geometry()
            edge.set_edge_name(from_node, to_node)
            edges_list.append(edge)


def  _edge_exists(edges_list, from_node, to_node):
    for edge in edges_list:
        if edge.are_nodes_in_edge(from_node.name, to_node.name):
            return True
    return False
    
def add_networkx_graph(nodes_list, edges_list):
    G = nx.Graph()
    for node in nodes_list:
        G.add_node(node.name)
    for edge in edges_list:
        G.add_edge(edge.from_node.name, edge.to_node.name, weight=edge.distance)
    return G


def main():
    # read the csv file as dataframe
    centroids_csv = '/home/blaxeep/workspace/od-dash/data/geodata_names/perif_centroids.csv'
    centroids_df = pd.read_csv(centroids_csv, sep='\t')
    nodes_list, edges_list, net_graph = read_adjacency_list_from_file(centroids_df)
    # convert dataframe contents to nodes and edges
    #return the result


if __name__ == '__main__':
    main()