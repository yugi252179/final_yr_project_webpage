@app.route("/")
def home():
    return render_template("dash.html")

@app.route("/brain_tumor.html")
def brain_tumor_page():
    return render_template("brain_tumor.html")

@app.route("/heart_prediction.html")
def heart_prediction_page():
    return render_template("heart_prediction.html")

@app.route("/eye_glaucoma.html")
def eye_glaucoma_page():
    return render_template("eye_glaucoma.html")
