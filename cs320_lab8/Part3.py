from flask import Flask, request

app = Flask(__name__)

data = {}

@app.route('/')
def lookup():
    key = request.args.get("key", "") # (1) query string
    value = data.get(key, "")
    print(data)
    return value + "\n" # (3) response body

@app.route('/', methods=["POST"])
def put():
    key = request.args.get("key") # (1) query string
    value = request.get_data(as_text=True) # (2) request body
    data[key] = value
    print(data)
    return "success\n" # (3) response body

if __name__ == '__main__':
    app.run(host="0.0.0.0")