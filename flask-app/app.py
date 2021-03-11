from keras.models import load_model
from flask import Flask
from flask import render_template
from flask import request

app = Flask("diabetes-prediction")

model = load_model('../dl-model/diabetes-model.h5')

model.summary()


@app.route("/")
def lw():
    '''
    p = model.predict([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
    print(round(p[0][0]))
    return("{}".format(round(p[0][0])))
    '''
    form = render_template("form.html")
    return form


@app.route("/predict")
def predict_value():
    lst = []

    for i in range(1, 9):
        lst.append(float(request.args.get(str(i))))

    p = model.predict([lst])
    print(round(p[0][0]))
    return("{}".format(round(p[0][0])))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
