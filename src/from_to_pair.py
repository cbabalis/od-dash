import osrm
import pdb


class FromToPair:
    """Class to define and save a path between two points
    """
    def __init__(self, from_node, to_node):
        self.from_node = FromToPair.Node(from_node)
        self.to_node = FromToPair.Node(to_node)
        self.geometry = None
    
    def compute_geometry(self):
        from_n = osrm.Point(latitude=self.from_node.lat,
                               longitude=self.from_node.lon)
        to_n = osrm.Point(latitude=self.to_node.lat,
                          longitude=self.to_node.lon)
        result = osrm.simple_route(from_n, to_n) #, output='route', overview="full", geometry='wkt')
        # https://github.com/ustroetz/python-osrm
        self.geometry = result['routes'][0]['geometry']
    
    def get_geometry():
        if not self.geometry:
            return -1
        else:
            return self.geometry
    
    class Node:
        def __init__(self, name):
            self.name = name
            self.lat = ''
            self.lon = ''
        
        def set_lat(self, lat):
            self.lat = lat
        
        def set_lon(self, lon):
            self.lon = lon
        
        def init_node(self, name, lon, lat):
            self.name = name
            self.lon = lon
            self.lat = lat
