from flask import Flask, request, jsonify
from views import views

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
