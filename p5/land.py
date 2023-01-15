#project: P5
#submitter: qyao34
#partner: sye22
import zipfile
import sqlite3
import numpy as np
import os
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

def open(name):
    return Connection(name)

class Connection:
    def __init__(self, name):
        self.db = sqlite3.connect(name+".db")
        self.zf = zipfile.ZipFile(name+".zip")
        
    def __enter__(self):
        return self
    
    def __exit__(self, type_usage, value, trace):
        self.close()
        
    def close(self):
        self.db.close()
        self.zf.close()
    
    def list_images(self):
        name = self.zf.namelist()
        name.sort()
        return name
    
    def image_year(self, name):
        cur = self.db.cursor()
        for row in cur.execute('SELECT * FROM images'):
            if row[1] == name: #https://docs.python.org/zh-cn/3/library/sqlite3.html
                year = int(row[0])
                pass
        return year
    
    def image_name(self, name):
        cur = self.db.cursor()
        for row in cur.execute('SELECT * FROM images'):
            if row[1] == name:
                image_id = row[2]
                pass
        for row in cur.execute('SELECT * FROM places'):
            if row[0] == image_id:
                image = row[1]
        return image
    
    def image_load(self, name):
        self.zf.extract(name)
        np_return = np.load(name)
        os.remove(name)
        return np_return
    
    def image_lat(self, name):
        cur = self.db.cursor()
        for row in cur.execute('SELECT * FROM images'):
            if row[1] == name:
                place_id = row[2]
                pass
        for row in cur.execute('SELECT * FROM places'):
            if row[0] == place_id:
                lat = row[2]
        return lat
    
    def lat_regression(self, use_code, ax = None):
        images_list = self.list_images()
        df_latperc = pd.DataFrame(columns=['latitute','percent'])
        i = 0
        for image_npy in images_list:
            if self.image_name(image_npy)[0:4] == 'samp':
                npy = self.image_load(image_npy)
                npy = npy - use_code
                df_latperc = df_latperc.append(pd.DataFrame({'latitute':[self.image_lat(image_npy)],'percent':[100*np.sum(npy == 0) / np.size(npy)]}), ignore_index = True)
        df_latperc_array = df_latperc.values
        x = df_latperc_array[:,0]
        y = df_latperc_array[:,1]
        #https://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html#sphx-glr-auto-examples-plot-isotonic-regression-py
        Linear_regree = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1).fit(x[:,np.newaxis],y)
        if ax == None:
            return (float(Linear_regree.coef_[0]),float(Linear_regree.intercept_))
        else:
            ax.scatter(x, y, s = 24)
            ax.plot(x ,Linear_regree.predict(x[:, np.newaxis]) ,linewidth = 2)
            return (float(Linear_regree.coef_[0]),float(Linear_regree.intercept_))
        
    def year_regression(self, name, list_code, ax = None):
        images_list = self.list_images()
        df_yearperc = pd.DataFrame(columns=['year','percent'])
        for image_npy in images_list:
            if self.image_name(image_npy) == name:
                npy = self.image_load(image_npy)
                freq_sum = 0
                for i in list_code:
                    freq_sum += np.sum((npy-i) == 0)
                df_yearperc = df_yearperc.append(pd.DataFrame({'year':[self.image_year(image_npy)],'percent':[100*freq_sum / np.size(npy)]}), ignore_index = False)
        df_yearperc_array = df_yearperc.values
        x = df_yearperc_array[:,0]
        y = df_yearperc_array[:,1]
        Linear_regree = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1).fit(x[:,np.newaxis],y)
        ax.scatter(x, y, s = 24)
        ax.plot(x ,Linear_regree.predict(x[:, np.newaxis]) ,linewidth = 2)
        return (float(Linear_regree.coef_[0]),float(Linear_regree.intercept_))
    
    def animate(self, name):
        images_list = self.list_images()
        year_list = list()
        for image_npy in images_list:
            if self.image_name(image_npy) == name:
                year_list.append(self.image_year(image_npy))
        year_list.sort()
        
        fig, ax = plt.subplots(figsize = (10,10))
        fps = 1
        
        use_cmap = np.zeros(shape=(256,4))
        use_cmap[:,-1] = 1
        uses = np.array([
            [0, 0.00000000000, 0.00000000000, 0.00000000000],
            [11, 0.27843137255, 0.41960784314, 0.62745098039],
            [12, 0.81960784314, 0.86666666667, 0.97647058824],
            [21, 0.86666666667, 0.78823529412, 0.78823529412],
            [22, 0.84705882353, 0.57647058824, 0.50980392157],
            [23, 0.92941176471, 0.00000000000, 0.00000000000],
            [24, 0.66666666667, 0.00000000000, 0.00000000000],
            [31, 0.69803921569, 0.67843137255, 0.63921568628],
            [41, 0.40784313726, 0.66666666667, 0.38823529412],
            [42, 0.10980392157, 0.38823529412, 0.18823529412],
            [43, 0.70980392157, 0.78823529412, 0.55686274510],
            [51, 0.64705882353, 0.54901960784, 0.18823529412],
            [52, 0.80000000000, 0.72941176471, 0.48627450980],
            [71, 0.88627450980, 0.88627450980, 0.75686274510],
            [72, 0.78823529412, 0.78823529412, 0.46666666667],
            [73, 0.60000000000, 0.75686274510, 0.27843137255],
            [74, 0.46666666667, 0.67843137255, 0.57647058824],
            [81, 0.85882352941, 0.84705882353, 0.23921568628],
            [82, 0.66666666667, 0.43921568628, 0.15686274510],
            [90, 0.72941176471, 0.84705882353, 0.91764705882],
            [95, 0.43921568628, 0.63921568628, 0.72941176471],
        ])
        for row in uses:
            use_cmap[int(row[0]),:-1] = row[1:]
        use_cmap = ListedColormap(use_cmap)
        
        def show_img(frame_num):
            ax.cla()
            for image_npy in images_list:
                if self.image_name(image_npy) == name and self.image_year(image_npy) == year_list[frame_num-1]:
                    map_np = self.image_load(image_npy)
                    pass
            ax.imshow(map_np, cmap = use_cmap, vmin=0, vmax=255)
            ax.set_title(str(year_list[frame_num-1]), fontsize= 24)
            return
        
        anim = FuncAnimation(fig, show_img, frames=len(year_list), interval = 1000/fps)
        plt.close(fig)
        
        html =anim.to_html5_video()
        return html