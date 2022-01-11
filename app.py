from flask import Flask, render_template, request, redirect, url_for,session
from werkzeug.utils import secure_filename
import os
from extract_audio import return_classified

app = Flask(__name__)
app.secret_key = "secret key"

@app.route('/', methods = ['GET', 'POST'])
def home():
    
    if request.method == 'POST':
        
        f = request.files['file']
       
        if ".wav" in f.filename:
            session['path'] = os.path.join(os.getcwd(), "Music", f"{secure_filename(f.filename)}")
            f.save(session['path'])
            session["filename"] = f.filename
            prediction = return_classified(session['path'])
            return render_template("index.html", message=f"Your file {session['filename']} is predicted to be of Genre {prediction}.")
        else:
            return render_template("index.html", message="You must enter .wav audio file.")
        
        
    else:
        return render_template("index.html", message=" ")



if __name__ == "__main__":
    app.run(debug = True)