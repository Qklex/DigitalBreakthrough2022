import os
import gc
import pandas as pd
import numpy as np
from flask import *
import pickle
from ast import literal_eval
from sklearn import preprocessing
import xgboost as xgb

ALLOWED_EXTENSIONS = set(['csv','xls','xlsx'])
app = Flask(__name__)

clf = pickle.load(open('model.pkl', 'rb'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/success', methods=['POST','GET'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            f.save("files/"+f.filename)
            return render_template("success.html", data=Markup(data("files/"+f.filename)))

    return render_template("failed.html")

@app.route('/download')
def downloadFile ():
    path = r'download_file/otchet.csv'
    return send_file(path, as_attachment=True)

def data(file):
    file_data = pd.read_csv(file)
    df = pd.DataFrame(file_data)
    df['Data1_int'] = df.apply(lambda row: literal_eval(row['Data']), axis=1)
    pred = df.copy()
    pred.drop(columns=pred.columns[1], axis=1, inplace=True)
    print(pred)
    pred.drop(columns=pred.columns[4], axis=1, inplace=True)
    print(pred)
    data_int = pred['Data1_int'].copy()
    pred.drop(columns=pred.columns[-1], axis=1, inplace=True)
    print(pred)

    temp_array = []
    for i in range(data_int.size):
        temp_array.append(data_int[i])

    data_1 = pd.DataFrame(temp_array, columns=colums(np_array=np.array(data_int[0])))
    print(data_1)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 2))

    names = data_1.columns
    d = scaler.fit_transform(data_1)

    scaled_df = pd.DataFrame(d, columns=names)
    pred.drop(columns=pred.columns[0], axis=1, inplace=True)
    pred = pred.join(scaled_df)
    prediction=clf.predict(pred)
    df['Class'] = prediction

    html = df.to_html()
    df['id,Class'] = df['id'].astype(str) + ',' + df['Class'].astype(str)

    colums_name=['id,Class']
    gc.collect()
    df.to_excel(r'download_file/otchet.xlsx',columns=colums_name,index=False)

    os.remove(file)
    return html


def colums(np_array):
    result = []
    for i in range(len(np_array)):
        result.append(str(i))
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4567)