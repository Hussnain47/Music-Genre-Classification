from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
pr = ""

@app.route('/', methods = ['GET', 'POST'])
def home():
    global pr
    print(request.method)
    if request.method == 'POST':
        print("A")
        f = request.files['file']
        print("B")
        print(f.filename)
        if ".wav" in f.filename:
            f.save(os.path.join(os.getcwd(), "Music", f"File 1 {secure_filename(f.filename)}"))
            pr = f.filename
        elif f.filename==pr:
            return render_template("index.html", message=" ")
        else:
            pr = f.filename
            return render_template("index.html", message="You must enter .wav file.")
        name = f.filename
        print(name)
        del f
        return render_template("index.html", message=f"Your file {name} has been saved.")
    else:
        return render_template("index.html", message=" ")


if __name__ == "__main__":
    app.run(debug = True)