from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

@app.before_request
def before_request():
    print("Antes de la petición")

@app.after_request
def after_request(response):
    print("Después de la petición")
    return response

@app.route("/")
def index():
    # return "¡Hola, Mundooooo!"
    return render_template("index.html")

@app.route("/contacto/<nombre>/<int:edad>")
def contacto(nombre, edad):
    data = {"titulo": "Contacto",
              "nombre": nombre,
              "edad": edad
              }
    return render_template("contacto.html", data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get("param1"))
    return "Ok"

def pagina_no_encontrada(error):
    return redirect (url_for("index"))

if __name__ == "__main__":
    app.add_url_rule('/query_string', view_func=query_string)
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug=True)
