from flask import Flask, request, jsonify, render_template, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
from io import BytesIO
import io
import pandas as pd
import os
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from flask_caching import Cache

app = Flask(__name__)

# Flask-Caching 설정
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

# MongoDB 설정
client = MongoClient('mongodb+srv://flaskuser:1111@flaskdb.le3ff4y.mongodb.net/?retryWrites=true&w=majority&appName=flaskDB')
db = client['testdb']
collection = db['testcollection']

# 메인 페이지
@app.route('/', methods=['GET'])
def main():
    return render_template('main.html')

# CSV 파일 업로드 페이지
@app.route('/data', methods=['GET'])
def upload_get():
    return render_template('upload.html')

# CSV 파일 DB 저장
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        data = pd.read_csv(file_stream)
        records = data.to_dict(orient='records')
        collection.insert_many(records)
        return jsonify({'message': 'File uploaded and data saved to MongoDB'}), 201

# 데이터 생성 (Create)
@app.route('/create', methods=['POST'])
def create():
    data = request.json
    result = collection.insert_one(data)
    return jsonify({'_id': str(result.inserted_id)}), 201

# 데이터 조회 (Read)
@app.route('/read', methods=['GET'])
def read_all():
    data = list(collection.find())
    for item in data:
        item['_id'] = str(item['_id'])
    return jsonify(data), 200

# 데이터 업데이트 (Update)
@app.route('/update/<id>', methods=['PUT'])
def update(id):
    data = request.json
    result = collection.update_one({'_id': ObjectId(id)}, {'$set': data})
    if result.modified_count > 0:
        return jsonify({'message': 'Data updated successfully'}), 200
    else:
        return jsonify({'error': 'Data not found'}), 404

# 데이터 삭제 (Delete)
@app.route('/delete/<id>', methods=['DELETE'])
def delete(id):
    result = collection.delete_one({'_id': ObjectId(id)})
    if result.deleted_count > 0:
        return jsonify({'message': 'Data deleted successfully'}), 200
    else:
        return jsonify({'error': 'Data not found'}), 404

# 전체 데이터 삭제 (Delete All)
@app.route('/delete_all', methods=['DELETE'])
def delete_all():
    result = collection.delete_many({})
    return jsonify({'message': 'All data deleted successfully'}), 200

# 데이터 시각화
@app.route('/analysis', methods=['GET'])
@cache.cached(timeout=300)  # Cache for 5 minutes
def analysis():
    data = pd.DataFrame(list(collection.find()))
    if not {'coil', 'magnet', 'wind', 'power'}.issubset(data.columns):
        return jsonify({'error': 'Required columns not found in data'}), 400

    # 데이터 샘플링 (전체 데이터의 10%만 사용)
    sample_data = data.sample(frac=0.1, random_state=42)

    sample_data = sample_data[['coil', 'magnet', 'wind', 'power']]

    # Calculate Pearson correlation coefficient
    corr = sample_data.corr(method='pearson')

    # Create scatter plot matrix
    sns.pairplot(sample_data)
    scatter_plot = BytesIO()
    plt.savefig(scatter_plot, format='png')
    scatter_plot.seek(0)
    scatter_plot_base64 = base64.b64encode(scatter_plot.getvalue()).decode('utf8')
    plt.clf()

    # Save correlation matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    heatmap = BytesIO()
    plt.savefig(heatmap, format='png')
    heatmap.seek(0)
    heatmap_base64 = base64.b64encode(heatmap.getvalue()).decode('utf8')
    plt.clf()

    return render_template('analysis.html', correlation_image=heatmap_base64, scatter_image=scatter_plot_base64)

# 예측 기능 추가
@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        coil = float(request.form['coil'])
        magnet = float(request.form['magnet'])
        wind = float(request.form['wind'])

        data = pd.DataFrame(list(collection.find()))
        if not {'coil', 'magnet', 'wind', 'power'}.issubset(data.columns):
            return jsonify({'error': 'Required columns not found in data'}), 400

        data = data[['coil', 'magnet', 'wind', 'power']]
        X = data[['coil', 'magnet', 'wind']]
        y = data['power']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 예측값 및 정확도 계산
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)

        prediction = model.predict([[coil, magnet, wind]])[0]

        return redirect(url_for('result', prediction=prediction, accuracy=accuracy))

    return render_template('model.html')

# 결과 페이지 출력
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    accuracy = request.args.get('accuracy')
    return render_template('result.html', prediction=prediction, accuracy=accuracy)

# 역설계 기능
@app.route('/retro', methods=['GET', 'POST'])
def retro():
    if request.method == 'POST':
        power = float(request.form['power'])

        data = pd.DataFrame(list(collection.find()))
        if not {'coil', 'magnet', 'wind', 'power'}.issubset(data.columns):
            return jsonify({'error': 'Required columns not found in data'}), 400

        data = data[['coil', 'magnet', 'wind', 'power']]
        X = data[['coil', 'magnet', 'wind']]
        y = data['power']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 피처 임포턴스를 기반으로 power를 반대로 예측하는 모델 생성
        inv_model = RandomForestRegressor(n_estimators=100, random_state=42)
        inv_model.fit(y_train.values.reshape(-1, 1), X_train)

        # 예측값 및 정확도 계산
        y_pred_inv = inv_model.predict(y_test.values.reshape(-1, 1))
        accuracy_inv = r2_score(X_test, y_pred_inv)

        prediction = inv_model.predict([[power]])
        coil, magnet, wind = prediction[0]

        return redirect(url_for('result_retro', coil=coil, magnet=magnet, wind=wind, accuracy=accuracy_inv))

    return render_template('retro.html')

# 역설계 결과 출력
@app.route('/result_retro')
def result_retro():
    coil = request.args.get('coil')
    magnet = request.args.get('magnet')
    wind = request.args.get('wind')
    accuracy = request.args.get('accuracy')
    return render_template('result_retro.html', coil=coil, magnet=magnet, wind=wind, accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
