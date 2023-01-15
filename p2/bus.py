from math import sin, cos, asin, sqrt, pi
from datetime import datetime
from zipfile import ZipFile
from graphviz import Graph, Digraph
from IPython.display import display
import pandas as pd

class BusDay:
    def __init__(self, date_time):
        weekday_str = str(date_time.strftime("%A")).lower()
        
        with ZipFile('mmt_gtfs.zip') as zf:
            #choose the information
            with zf.open("calendar.txt") as f:
                self.df_calendar = pd.read_csv(f)
            with zf.open("trips.txt") as f:
                self.df_trips = pd.read_csv(f)
            with zf.open("stops.txt") as f:
                self.df_stops = pd.read_csv(f)
            with zf.open("stop_times.txt") as f:
                self.df_stop_times = pd.read_csv(f)
            
        self.df_calendaronday = self.df_calendar[self.df_calendar[weekday_str] == 1]#https://blog.csdn.net/damei2017/article/details/84959895
        
        date_time_int = int(date_time.strftime("%Y%m%d"))#pick the calendar rows have required time
        self.df_calendaronday_ava = self.df_calendaronday[self.df_calendaronday['start_date'] < date_time_int]
        self.df_calendaronday_ava = self.df_calendaronday_ava[self.df_calendaronday_ava['end_date'] > date_time_int]
        
        self.service_ids = self.df_calendaronday_ava['service_id'].tolist()#http://sofasofa.io/forum_main_post.php?postid=1002397
        
        df_trips_ava = self.df_trips[self.df_trips['service_id'].isin(self.df_calendaronday_ava['service_id'])]
        df_stop_times_ava = self.df_stop_times[self.df_stop_times['trip_id'].isin(df_trips_ava['trip_id'])]
        self.df_stops_ava = self.df_stops[self.df_stops['stop_id'].isin(df_stop_times_ava['stop_id'])]
        
        df_stops_loc = pd.DataFrame(columns=['x', 'y'])
        for i in range(len(self.df_stops_ava)):
            loc = Location(latlon = (self.df_stops_ava.iloc[i]['stop_lat'], self.df_stops_ava.iloc[i]['stop_lon']))
            df_stops_loc = df_stops_loc.append(pd.DataFrame({'x':[loc.x],'y':[loc.y]}),ignore_index=True)
        
        df_stops_loc.index = self.df_stops_ava.index
        self.df_stops_ava['x'] = df_stops_loc['x']
        self.df_stops_ava['y'] = df_stops_loc['y']
        
        self.BST_stops = BST(self.df_stops_ava)
        
        for i in range(6):
            self.BST_stops.add()
            
    def get_trips(self, route_short_name = None):
        if (route_short_name == None):
            df_avaliabletrips =self.df_trips
        else:
            df_avaliabletrips = self.df_trips[self.df_trips['route_short_name'] == route_short_name]
        #print(df_avaliabletrips)
        df_tripsonday = df_avaliabletrips[df_avaliabletrips['service_id'].isin(self.df_calendaronday_ava['service_id'])]
        #https://blog.csdn.net/qq1483661204/article/details/79824381
        df_tripsonday_sort = df_tripsonday.sort_values('trip_id', ascending = True, inplace = False)
        trip_object_list = list()
        for i in range(len(df_tripsonday_sort)):
            trip_object_list.append(Trip(df_tripsonday_sort.iloc[i]['trip_id'], df_tripsonday_sort.iloc[i]['route_short_name'], df_tripsonday_sort.iloc[i]['bikes_allowed']))
        return trip_object_list
    
    def get_stops(self):
        df_stops_sort = self.df_stops_ava.sort_values('stop_id', ascending = True, inplace = False)
        stop_object_list = list()
        for i in range(len(df_stops_sort)):
            Locat = Location(latlon = (df_stops_sort.iloc[i]['stop_lat'], df_stops_sort.iloc[i]['stop_lon']))
            stop_object_list.append(Stop(df_stops_sort.iloc[i]['stop_id'], Locat, df_stops_sort.iloc[i]['wheelchair_boarding']))
        return stop_object_list
    
    def get_stops_rect(self, X, Y):
        self.BST_stops.stops_search_rect(X, Y)
        
        df_stops_sort = self.BST_stops.stops_searched_rect.sort_values('stop_id', ascending = True, inplace = False)
        stop_object_list = list()
        
        for i in range(len(df_stops_sort)):
            Locat = Location(latlon = (df_stops_sort.iloc[i]['stop_lat'], df_stops_sort.iloc[i]['stop_lon']))
            stop_object_list.append(Stop(df_stops_sort.iloc[i]['stop_id'], Locat, df_stops_sort.iloc[i]['wheelchair_boarding']))
        return stop_object_list
    
    def get_stops_circ(self, Loc, radius):
        X = (Loc[0]-radius, Loc[0]+radius)
        Y = (Loc[1]-radius, Loc[1]+radius)
        
        self.BST_stops.stops_search_rect(X, Y)
        
        df_stops_sort = self.BST_stops.stops_searched_rect.sort_values('stop_id', ascending = True, inplace = False)
        stop_object_list = list()
        
        for i in range(len(df_stops_sort)):
            Locat = Location(latlon = (df_stops_sort.iloc[i]['stop_lat'], df_stops_sort.iloc[i]['stop_lon']))
            if ((Locat.x-Loc[0])**2 + (Locat.y-Loc[1])**2) <= radius**2:
                stop_object_list.append(Stop(df_stops_sort.iloc[i]['stop_id'], Locat, df_stops_sort.iloc[i]['wheelchair_boarding']))
        return stop_object_list
    
    def scatter_stops(self, Ax):
        df_stops_ava_wheel = self.df_stops_ava[self.df_stops_ava['wheelchair_boarding'] == 1]
        df_stops_ava_wheeln = self.df_stops_ava[self.df_stops_ava['wheelchair_boarding'] == 0]
        df_stops_ava_wheel.plot.scatter('x','y', ax = Ax, color = "red")
        df_stops_ava_wheeln.plot.scatter('x','y', ax = Ax, color = "0.7")
        Ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        Ax.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        Ax.spines["top"].set_visible(False)
        Ax.spines["right"].set_visible(False)
        #https://blog.csdn.net/sinat_34328764/article/details/80246139
        Ax.xaxis.set_ticks_position('bottom')   
        Ax.yaxis.set_ticks_position('left')
        Ax.spines['bottom'].set_position(('data', 0))
        Ax.spines['left'].set_position(('data', 0))
        return None

    def draw_tree(self, Ax):
        self.BST_stops.sec_line()
        line_data = self.BST_stops.line_df
        for i in range(len(line_data)):
            if line_data.iloc[i]['direction'] == 'x':
                Ax.plot((line_data.iloc[i]['position'], line_data.iloc[i]['position']), (line_data.iloc[i]['lim'][0], line_data.iloc[i]['lim'][1]), lw=10/line_data.iloc[i]['level'], color="yellow")
            else:
                Ax.plot((line_data.iloc[i]['lim'][0], line_data.iloc[i]['lim'][1]), (line_data.iloc[i]['position'], line_data.iloc[i]['position']), lw=10/line_data.iloc[i]['level'], color="yellow")
        Ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        Ax.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        Ax.spines["top"].set_visible(False)
        Ax.spines["right"].set_visible(False)
        #https://blog.csdn.net/sinat_34328764/article/details/80246139
        Ax.xaxis.set_ticks_position('bottom')   
        Ax.yaxis.set_ticks_position('left')
        Ax.spines['bottom'].set_position(('data', 0))
        Ax.spines['left'].set_position(('data', 0))
        return None

class Trip:
    def __init__(self, trip_id, route_short_name, bikes_allowed):
        self.trip_id = trip_id
        self.route_short_name = route_short_name
        self.bikes_allowed = bikes_allowed > 0

    def __repr__(self):
        #https://python3-cookbook.readthedocs.io/zh_CN/latest/c02/p15_interpolating_variables_in_strings.html
        return str('Trip({trip_id}, {route_short_name}, {bikes_allowed})'.format(trip_id = self.trip_id, route_short_name = self.route_short_name, bikes_allowed = self.bikes_allowed))

class Stop():
    def __init__(self, stop_id, location, wheelchair_boarding):
        self.stop_id = stop_id
        self.location = location
        self.wheelchair_boarding = wheelchair_boarding > 0
    
    def __repr__(self):
        return str('Stop({stop_id}, {location}, {wheelchair_boarding})'.format(stop_id = self.stop_id, location = self.location, wheelchair_boarding = self.wheelchair_boarding))

class Node:
    def __init__(self, direction = '', val = 0, level = 1 ,key = None, contour = None):
        self.direction = direction
        self.val = val
        self.level = level
        self.key = key
        self.contour = contour
        self.left = None
        self.right = None
        
    def dump(self):
        print(' '* self.level + self.direction)
        if self.left !=None and self.right != None: 
            self.left.dump()
            self.right.dump()
class BST():
    def __init__(self, df_stops_ava):
        contour = pd.DataFrame({'x':[-8, -8, 8, 8],'y':[8, -8, 8, -8]})
        contour.index = ['lt', 'lb', 'rt', 'rb']
        self.root = Node('x' , 0, 1, df_stops_ava, contour)
        self.size = 1
        
    def add(self):
        start = self.root
        self.add_node(start)
        self.size += 1
        
    def add_node(self, node):
        contour_child_left = pd.DataFrame(columns = node.contour.columns, index = node.contour.index)
        contour_child_right = pd.DataFrame(columns = node.contour.columns, index = node.contour.index)
        if node.left == None and node.right == None:
            if (self.size % 2) != 0:
                df_stops_ava_sort = node.key.sort_values('x', ascending = True, inplace = False)
                
                node.dirction = 'x'
                node.val = (df_stops_ava_sort.iloc[len(df_stops_ava_sort)//2]['x']+df_stops_ava_sort.iloc[len(df_stops_ava_sort)//2+1]['x'])/2
                
                key_left = df_stops_ava_sort[0:len(df_stops_ava_sort)//2+1]
                contour_child_left.loc['lt'] = node.contour.loc['lt']
                contour_child_left.loc['lb'] = node.contour.loc['lb']
                contour_child_left.iloc[2]['y'] = node.contour.iloc[2]['y']
                contour_child_left.iloc[3]['y'] = node.contour.iloc[3]['y']
                contour_child_left.iloc[2]['x'] = node.val
                contour_child_left.iloc[3]['x'] = node.val
                node.left = Node('y', 0, self.size+1, key_left, contour_child_left)
                
                key_right = df_stops_ava_sort[len(df_stops_ava_sort)//2+1:len(df_stops_ava_sort)]
                contour_child_right.iloc[0]['y'] = node.contour.iloc[0]['y']
                contour_child_right.iloc[1]['y'] = node.contour.iloc[1]['y']
                contour_child_right.iloc[0]['x'] = node.val
                contour_child_right.iloc[1]['x'] = node.val
                contour_child_right.loc['rt'] = node.contour.loc['rt']
                contour_child_right.loc['rb'] = node.contour.loc['rb']
                node.right = Node('y', 0, self.size+1, key_right, contour_child_right)
            else:
                df_stops_ava_sort = node.key.sort_values('y', ascending = True, inplace = False)
                
                node.dirction = 'y'
                node.val = (df_stops_ava_sort.iloc[len(df_stops_ava_sort)//2]['y']+df_stops_ava_sort.iloc[len(df_stops_ava_sort)//2+1]['y'])/2
                
                key = df_stops_ava_sort[0:len(df_stops_ava_sort)//2+1]
                contour_child_left.loc['lb'] = node.contour.loc['lb']
                contour_child_left.loc['rb'] = node.contour.loc['rb']
                contour_child_left.iloc[0]['y'] = node.val
                contour_child_left.iloc[2]['y'] = node.val
                contour_child_left.iloc[0]['x'] = node.contour.iloc[0]['x']
                contour_child_left.iloc[2]['x'] = node.contour.iloc[2]['x']
                node.left = Node('x', 0, self.size+1, key, contour_child_left)
                
                key = df_stops_ava_sort[len(df_stops_ava_sort)//2+1:len(df_stops_ava_sort)]
                contour_child_right.loc['lt'] = node.contour.loc['lt']
                contour_child_right.loc['rt'] = node.contour.loc['rt']
                contour_child_right.iloc[1]['y'] = node.val
                contour_child_right.iloc[3]['y'] = node.val
                contour_child_right.iloc[1]['x'] = node.contour.iloc[1]['x']
                contour_child_right.iloc[3]['x'] = node.contour.iloc[3]['x']
                node.right = Node('x', 0, self.size+1, key, contour_child_right)
            return
        
        else:
            self.add_node(node.left)
            self.add_node(node.right)
                
    def __len__(self):
        return self.size
                
    def sec_line(self):
        start = self.root
        self.line_df = pd.DataFrame(columns=['direction', 'position', 'level', 'lim'])
        self.node_line(start)
        
    def node_line(self, node):
        if node.left == None and node.right == None:
            return
        else:
            if  node.direction== 'x':
                #https://amberwest.github.io/2019/03/04/%E6%B7%BB%E5%8A%A0%E4%B8%80%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%88%B0DataFrame/
                self.line_df = self.line_df.append({'direction': 'x', 'position': node.val, 'level': node.level, 'lim': (node.contour.iloc[3]['y'], node.contour.iloc[2]['y'])}, ignore_index=True)
                
            if node.direction == 'y':
                self.line_df = self.line_df.append({'direction': 'y', 'position': node.val, 'level': node.level, 'lim': (node.contour.iloc[0]['x'], node.contour.iloc[2]['x'])}, ignore_index=True)
                
        self.node_line(node.left)
        self.node_line(node.right)
         
    def stops_search_rect(self, X, Y):
        start = self.root
        self.stops_searched_rect = pd.DataFrame(columns=self.root.key.columns)
        self.stops_search_node(start, X, Y)
        return self.stops_searched_rect
    
    def stops_search_node(self, node, X, Y):
        if node.left == None and node.right == None:
            stops_found = node.key[node.key['x'] <= X[1]]
            stops_found = stops_found[stops_found['x'] >= X[0]]
            stops_found = stops_found[stops_found['y'] <= Y[1]]
            stops_found = stops_found[stops_found['y'] >= Y[0]]
            self.stops_searched_rect = self.stops_searched_rect.append(stops_found, ignore_index=True)
        else:
            if node.direction == 'x':
                if node.val > X[1]:
                    self.stops_search_node(node.left, X, Y)
                if node.val < X[0]:
                    self.stops_search_node(node.right, X, Y)
                if node.val <= X[1] and node.val >= X[0]:
                    self.stops_search_node(node.left, X, Y)
                    self.stops_search_node(node.right, X, Y)
                    
            if node.direction == 'y':
                if node.val > Y[1]:
                    self.stops_search_node(node.left, X, Y)
                if node.val < Y[0]:
                    self.stops_search_node(node.right, X, Y)
                if node.val <= Y[1] and node.val >= Y[0]:
                    self.stops_search_node(node.left, X, Y)
                    self.stops_search_node(node.right, X, Y)
                    
def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculates the distance between two points on earth using the
    harversine distance (distance between points on a sphere)
    See: https://en.wikipedia.org/wiki/Haversine_formula

    :param lat1: latitude of point 1
    :param lon1: longitude of point 1
    :param lat2: latitude of point 2
    :param lon2: longitude of point 2
    :return: distance in miles between points
    """
    lat1, lon1, lat2, lon2 = (a/180*pi for a in [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * asin(min(1, sqrt(a)))
    d = 3956 * c
    return d


class Location:
    """Location class to convert lat/lon pairs to
    flat earth projection centered around capitol
    """
    capital_lat = 43.074683
    capital_lon = -89.384261

    def __init__(self, latlon=None, xy=None):
        if xy is not None:
            self.x, self.y = xy
        else:
            # If no latitude/longitude pair is given, use the capitol's
            if latlon is None:
                latlon = (Location.capital_lat, Location.capital_lon)

            # Calculate the x and y distance from the capital
            self.x = haversine_miles(Location.capital_lat, Location.capital_lon,
                                     Location.capital_lat, latlon[1])
            self.y = haversine_miles(Location.capital_lat, Location.capital_lon,
                                     latlon[0], Location.capital_lon)

            # Flip the sign of the x/y coordinates based on location
            if latlon[1] < Location.capital_lon:
                self.x *= -1

            if latlon[0] < Location.capital_lat:
                self.y *= -1

    def dist(self, other):
        """Calculate straight line distance between self and other"""
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return "Location(xy=(%0.2f, %0.2f))" % (self.x, self.y)