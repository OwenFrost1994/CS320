import pandas as pd
from pandas import DataFrame, read_csv
from flask import Flask, request, jsonify
import re
import requests
import json
app = Flask(__name__)

visit_counts = {}


def visitor_counts(fn):
    visit_counts[fn.__name__] = 0
    visit_counts['A'] = 0
    visit_counts['B'] = 0
    def add():
        print(visit_counts)
        visit_counts[fn.__name__] += 1
        if request.args.get('from') == None:
            pass
        else:
            key = request.args['from']
            #if key not in visit_counts.keys():
            #    visit_counts[key] = 1
            #else:
            visit_counts[key] += 1
        return fn()
    add.__name__ = fn.__name__
    return add

global visit_home
global versionA
global versionB
visit_home = 0
versionA = 0
versionB = 0

@app.route('/')
@visitor_counts
def home():
    #global visit_home
    #visit_home += 1
    #print('home' + str(visit_home))
    with open("index.html") as f:
        html = f.read()
    #if visit_home < 10:
        #if visit_home % 2 != 0:
    if visit_counts['home'] <= 10:
        if visit_counts['home'] % 2 != 0:
            #print('Ahome' + str(visit_counts['home']))
            return html + "<h1><a href=\"donate.html?from=A\">Donate us here</a></h1>"
        else:
            #print('Bhome' + str(visit_counts['home']))
            return html + "<h1><a href=\"donate.html?from=B\">Donate us here</a></h1>"
    else:
        #if versionA >= versionB:
        if visit_counts['A'] > visit_counts['B']:
            return html + "<h1><a href=\"donate.html?from=A\">Donate us here</a></h1>"
        else:
            return html + "<h1><a href=\"donate.html?from=B\">Donate us here</a></h1>"

@app.route('/email', methods=["POST"])
@visitor_counts
def email():
    email = str(request.data, "utf-8")
    if re.match(r"[\w]+\@[\w]+\.[\w]+", email):
        with open("emails.txt", "a") as f: # open file in append mode
            f.write(email)
            f.write('\r\n')
        return jsonify("thanks")
    return jsonify("The email address is invalid")

@app.route('/browse.html')
@visitor_counts
def browse():
    df = pd.read_csv("main.csv")
    #df = df.round(6)
    #https://cloud.tencent.com/developer/ask/217284
    #https://www.cnblogs.com/liminghui3/p/8202653.html
    with open('browse.html') as f:
        html = f.read()
    html = html+df.to_html(header="true", table_id="table")
    return html

@app.route('/donate.html')
@visitor_counts
def donate():
    global versionA
    global versionB
    #print(request.args)
    if 'from' in request.args:
        val = str(request.args['from'])
        #print(val)
        #if val == 'A':
            #versionA += 1
            #print('A' + str(versionA))
        #else:
            #versionB += 1
            #print('B' + str(versionB))
    with open('donate.html') as f:
        html = f.read()
    return html

@app.route('/api.html')
@visitor_counts
def api():
    with open('api.html') as f:
        html = f.read()
    
    return html

@app.route('/listofrows.json')
@visitor_counts
def listofrows():
    df = pd.read_csv("main.csv")
    #print(request.args)
    if request.args == dict():
        #print('2')
        df_out = list()
        for i in range(len(df)):
            row = df.loc[i]
            df_out.append(json.loads(row.to_json(orient='index')))
        #df_out = df.to_json(orient='index')
        return jsonify(df_out)
    
    elif 'row' in request.args:
        #print('1')
        rownum = int(request.args['row'])
        df_out = df.loc[rownum]
        df_out = json.loads(df_out.to_json(orient='index'))
        return df_out
    else:
        #print('3')
        for i in request.args.keys():
            val=request.args.get(i)
            #print(type(val))
            limit = val.replace("(", "")
            #limit = request.args[i]
            #limit = limit.replace("(", "")
            limit = limit.replace(")", "")
            inter = limit.split(",")
            if len(inter) == 1:
                df_f = df[df[i] == float(inter[0])]
            else:
                df_f = df[df[i] >= float(inter[0])]
                df_f = df_f[df_f[i] <= float(inter[1])]
        df_out = list()
        for i in range(len(df_f)):
            #print(i)
            row = df_f.iloc[i]
            df_out.append(json.loads(row.to_json(orient='index')))
        return jsonify(df_out)

if __name__ == '__main__':
    app.run(host="0.0.0.0") # don't change this line!
