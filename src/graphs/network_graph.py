"""Module to represent a network as a graph.
"""

import osrm
import polyline


class Vertex:
    def __init__(self, name='', lat='', lon=''):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.neighbors_distance = {}
        self.neighbors_duration = {}
        self.total_weight = 0
    
    def set_lat(self, lat):
        self.lat = lat
    
    def set_lon(self, lon):
        self.lon = lon
    
    def set_new_neighbor_dist(self, name, weight):
        """ Method to set a new neighbor to the vertex.
        Args:
            name (str): name of the new neighbor
            weight (int): weight (or cost) for getting from the vertex to
                this neighbor.
        """
        self.neighbors_distance[name] = weight
    
    
    def set_new_neighbor_dur(self, name, weight):
        self.neighbors_duration[name] = weight


class Edge:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.geometry = None
        self.distance = 0
        self.duration = 0
        self.edge_name = self.set_edge_name(from_node, to_node)
        self.usage_weight = 0
    
    def compute_geometry(self):
        from_n = osrm.Point(latitude=self.from_node.lat,
                               longitude=self.from_node.lon)
        to_n = osrm.Point(latitude=self.to_node.lat,
                          longitude=self.to_node.lon)
        result = osrm.simple_route(from_n, to_n) #, output='route', overview="full", geometry='wkt')
        # also assign duration and distance
        self.distance = result['routes'][0]['distance']
        self.duration = result['routes'][0]['duration']
        # https://github.com/ustroetz/python-osrm
        self.geometry = result['routes'][0]['geometry']
    
    def get_geometry(self):
        self.compute_geometry()
        if not self.geometry:
            print("No geometry exists!")
            return -1
        else:
            return self.geometry
    
    def set_edge_name(self, from_node, to_node):
        """Method to set the name of an edge.

        Args:
            from_node (Node): [description]
            to_node (Node): [description]
        """
        self.edge_name = str(from_node.name) + str("-") + str(to_node.name)
    
    def are_nodes_in_edge(self, from_node_name, to_node_name):
        """Method to check if from and to nodes are in this edge

        Args:
            from_node_name ([type]): [description]
            to_node_name ([type]): [description]
        """
        if from_node_name not in self.edge_name:
            return False
        elif to_node_name not in self.edge_name:
            return False
        return True
    
    def add_to_usage_weight(self, weight):
        self.usage_weight += weight
    
    
    def get_geometry_lats_lons_lists(self):
        self.edge_lat_list = []
        self.edge_lon_list = []
        self.compute_geometry()
        try:
            route_lines = polyline.decode(self.geometry)
            #print("length of route lines points is ", len(route_lines))
        except:
            print("error in geometry of edge ", self.edge_name)
            return False
        for line in route_lines:
            lat, lon = line
            self.edge_lat_list.append(lat)
            self.edge_lon_list.append(lon)
        return (self.edge_lat_list, self.edge_lon_list)
    
    def is_node_in_edge(self, node_name):
        if node_name in self.edge_name:
            return True
        return False