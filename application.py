from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hey!"

# @app.route("/")
# def home():
#     return render_template("home.html")
