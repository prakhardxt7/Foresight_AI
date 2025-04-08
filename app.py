from flask import Flask, render_template, request, redirect, url_for, flash, session
from router.smart_query_router import detect_and_route
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = 'foresight-secret'  # 🔐 Use env variable in production
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 🔐 In-memory demo users
users = {
    "admin@foresight.ai": "123456",
    "nykaa@insights.com": "nykaa2024"
}

@app.route("/")
def home():
    return render_template("index.html", user=session.get("user"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if users.get(email) == password:
            session["user"] = email
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("login"))

@app.route("/about")
def about():
    return render_template("about.html", user=session.get("user"))

@app.route("/query", methods=["POST"])
def handle_query():
    user_query = request.form["query"]
    result = detect_and_route(user_query)
    return render_template("result.html", result=result, query=user_query, user=session.get("user"))

if __name__ == "__main__":
    app.run(debug=True)
