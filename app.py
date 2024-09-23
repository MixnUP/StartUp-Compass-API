from flask import Flask
from views import views
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.register_blueprint(views, url_prefix="/")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


# USE pytrends==4.9.2