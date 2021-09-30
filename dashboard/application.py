from flask import Flask, render_template, request


app = Flask(
    __name__, static_url_path="", static_folder="templates", template_folder="templates"
)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run("127.0.0.1")
