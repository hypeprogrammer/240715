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

import urllib.parse  # for encoding the query parameters
import ast

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
    if not {'Input Voltage, V', 'Frequancy, hz', 'Coil turns', 'Efficiency, %', 'rated power, W '}.issubset(data.columns):
        return jsonify({'error': 'Required columns not found in data'}), 400

    # 데이터 샘플링 (전체 데이터의 10%만 사용)
    sample_data = data.sample(frac=0.1, random_state=42)

    sample_data = sample_data[['Input Voltage, V', 'Frequancy, hz', 'Coil turns', 'Efficiency, %', 'rated power, W ']]


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
    plt.figure(figsize=(16, 12))
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
    # Define the columns A to Y based on your dataset
    required_columns = {
        'Input Voltage, V', 'Frequancy, hz', 'Outer diameter of stator , mm',
        'inner diameter of stator , mm', 'Length of Stator Core, mm',
        'Stacking Factor of Stator Core', 'Number of Stator Slots', 'Hs0, mm',
        'Hs1, mm', 'Hs2, mm', 'Bs1, mm', 'Bs2, mm', 'Coil turns', 'Coil pitch',
        'Wire area, mm^2', 'Number of Poles', 'Outer Diameter of Rotor, mm',
        'Inner Diameter of Rotor, mm', 'Length of Rotor Core, mm',
        'Stacking Factor of Rotor Core', 'Magnet Skew Width', 'Magnet Embrace',
        'Magnet Radian Length, mm', 'Magnet Axial Thickness, mm', 'Air gap, mm'
    }

    if request.method == 'POST':
        # Extract input features from the form
        try:
            input_data = {key: float(request.form[key]) for key in required_columns}
        except KeyError as e:
            return jsonify({'error': f'Missing input for required field: {e.args[0]}'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid input type. Please ensure all inputs are numeric.'}), 400

        # Load data from the database
        data = pd.DataFrame(list(collection.find()))

        if data.empty:
            return jsonify({'error': 'No data found in the database.'}), 400

        if not required_columns.issubset(data.columns):
            return jsonify({'error': 'Required columns not found in data'}), 400

        # Use columns A to Y as input features
        X = data[list(required_columns)]
        # Use columns Z onwards as target variables
        y = data.iloc[:, len(required_columns):]

        # Check if target variables are present
        if y.empty:
            return jsonify({'error': 'No target columns found for prediction.'}), 400

        # Train models for each target variable
        predictions = {}
        accuracies = {}

        # Iterate over each target column and predict
        for column in y.columns:
            model = RandomForestRegressor(n_estimators=1000, max_depth=100, random_state=42)
            model.fit(X, y[column])
            y_pred = model.predict(X)  # Predict using the same training data (overfitting)
            accuracy = r2_score(y[column], y_pred)
            predictions[column] = model.predict([list(input_data.values())])[0]
            accuracies[column] = accuracy

        # Redirect to result page with predictions and accuracies
        return redirect(url_for('result', predictions=predictions, accuracies=accuracies))

    return render_template('model.html')

# 결과 페이지 출력
@app.route('/result')
def result():
    # Decode the query parameters back into dictionaries
    predictions = ast.literal_eval(urllib.parse.unquote(request.args.get('predictions')))
    accuracies = ast.literal_eval(urllib.parse.unquote(request.args.get('accuracies')))

    # Convert accuracy values to percentage
    accuracies_percentage = {key: value * 100 for key, value in accuracies.items()}

    # Pass the decoded dictionaries to the template
    return render_template('result.html', predictions=predictions, accuracies=accuracies_percentage)



# 역설계 기능
@app.route('/retro', methods=['GET', 'POST'])
def retro():
    if request.method == 'POST':
        # Extract the target values from the form
        target_values = [float(request.form['Efficiency']),
                         float(request.form['Needed current']),
                         float(request.form['rated power']),
                         float(request.form['Slot Area']),
                         float(request.form['Slot Fill Factor']),
                         float(request.form['Stator Net Weight']),
                         float(request.form['Rotor Net Weight'])]

        # Load the data from the MongoDB collection
        data = pd.DataFrame(list(collection.find({}, {'_id': 0})))

        # Ensure that the dataset has the necessary columns
        if data.empty or len(data.columns) < 32:
            return jsonify({'error': 'Dataset does not contain the required columns'}), 400

        # Define input features (columns 26 to 32) and target variables (columns 1 to 25)
        X = data.iloc[:, 25:32]  # Columns 26 to 32 (index 25 to 31)
        y = data.iloc[:, :25]  # Columns 1 to 25 (index 0 to 24)

        # Train the model with the entire dataset to overfit
        inv_model = RandomForestRegressor(n_estimators=1000, max_depth=100, random_state=42)
        inv_model.fit(X, y)

        # Calculate the inverse prediction on the training data (overfitting)
        y_pred_inv = inv_model.predict(X)
        accuracy_inv = r2_score(y, y_pred_inv)

        # Use the model to predict the original features based on the provided target values
        prediction = inv_model.predict([target_values])

        # Extract predictions with actual column names
        predictions = {col: pred for col, pred in zip(y.columns, prediction[0])}

        # Redirect to the result page with predictions and accuracy
        return redirect(url_for('result_retro', predictions=predictions, accuracy=accuracy_inv))

    return render_template('retro.html')


# Result page for retro
@app.route('/result_retro')
def result_retro():
    # Retrieve predictions and accuracy from query parameters
    predictions = request.args.get('predictions')
    accuracy = request.args.get('accuracy')

    # Convert the string of predictions back to a dictionary
    predictions = ast.literal_eval(predictions)

    # Assuming the same accuracy for all features (this might be the case if it's a single overall model accuracy)
    accuracies = {key: float(accuracy) * 100 for key in predictions.keys()}  # Convert to percentage

    # Pass the decoded dictionaries to the template
    return render_template('result_retro.html', predictions=predictions, accuracies=accuracies)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)