from flask import Flask
from pipeline import main

app = Flask(__name__)


@app.route('/')
def hello_world():

    flow = main.build_flow()
    flow.run()

    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
