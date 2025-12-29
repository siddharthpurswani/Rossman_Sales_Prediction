from flask import Flask, request, jsonify, render_template
import os
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)

model = joblib.load('rossmann_model.joblib')



scaler = joblib.load('scaled.joblib')


model_columns = joblib.load("model_columns.joblib") 

def preprocessing(data, model, scaler):
    """Exact replica of Colab preprocessing - production ready"""
    
    # Step 1: Create single-row DataFrame with safe defaults
    predict_data = pd.DataFrame([data]).fillna(0)
    
    
    # Step 2: Categorical mappings (EXACT Colab logic)
    predict_data['Assortment'] = predict_data['Assortment'].map({'a': 1, 'b': 3, 'c': 2}).fillna(1)
    predict_data['StoreType'] = predict_data['StoreType'].apply(lambda x: 1 if x == 'b' else 0)
    predict_data['Year'] = predict_data['Year'].map({2013: 0, 2014: 1, 2015: 2}).fillna(0)
    
    # Step 3: Create dummies EXACTLY like training
    df_dummies_day = pd.get_dummies(predict_data['DayOfWeek'], prefix='DayOfWeek')
    df_dummies_month = pd.get_dummies(predict_data['month'], prefix='Month')
    df_dummies_store = pd.get_dummies(predict_data['Store'], prefix='Store')
    
    # Concatenate ALL features
    predict_data = pd.concat([predict_data, df_dummies_day, df_dummies_month, df_dummies_store], axis=1)
    
    # Drop original categorical columns
    predict_data.drop(columns=['DayOfWeek', 'month', 'Store'], inplace=True, errors='ignore')
    
    # Step 4: Scale numeric columns (SAFE version)
    numeric_cols = ['CompetitionDistance', 'cumm_sum', 'Assortment', 'Year']
    scaled_cols = ['scaled_CompetitionDistance', 'scaled_cumm_sum', 'scaled_assortment', 'scaled_year']
    
    # Only scale columns that exist
    available_numeric = [col for col in numeric_cols if col in predict_data.columns]
    print(f"🔍 DEBUG: Scaling {len(available_numeric)}/{len(numeric_cols)} numeric cols")
    
    if available_numeric and scaler:
        # Fill NaN → Scale → Assign back
        scale_data = predict_data[available_numeric].fillna(0)
        scaled_values = scaler.transform(scale_data)
        
        # Create scaled columns
        for i, col in enumerate(scaled_cols[:len(available_numeric)]):
            predict_data[col] = scaled_values[:, i]
    
    # Drop original numeric + problematic columns
    predict_data.drop(columns=numeric_cols + ['scaled_cumm_sum'], inplace=True, errors='ignore')
    
    # Step 5: BULLETPROOF column alignment (matches training EXACTLY)
    print(f"🔍 DEBUG: Before alignment: {predict_data.shape} cols")
    
    # Add missing columns as 0
    missing_cols = set(model_columns) - set(predict_data.columns)
    for col in missing_cols:
        predict_data[col] = 0

    # 3. Drop extra columns (optional but recommended)
    extra_cols = set(predict_data.columns) - set(model_columns)
    predict_data = predict_data.drop(columns=extra_cols)

    # 4. Reorder EXACTLY like training
    predict_data = predict_data[model_columns].fillna(0)
    
    print(f"🔍 DEBUG: Final shape: {predict_data.shape} (expected: {len(model_columns)} cols)")
    print(f"🔍 DEBUG: First 5 cols: {predict_data.columns[:5].tolist()}")
    
    # Step 6: Convert to numpy (sklearn requirement)
    X = predict_data.values
    
    # Predict
    y_pred = model.predict(X)[0]
    
    print(f"🔍 DEBUG: Prediction: {y_pred}")
    return float(y_pred)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    try:
        data = request.json['data']
        print("📥 Input:", data)
        prediction = preprocessing(data,model,scaler)
        print("✅ Prediction:", prediction)
        
        return jsonify({
            'predicted_sales': float(prediction),
            'store_id': int(data['Store'])
        })
    except Exception as e:
        print("❌ ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400
    
@app.route('/features')
def features():
    return jsonify({
        'total_features': len(model_columns),
        'first_10': model_columns[:10],
        'last_10': model_columns[-10:],
        'sample': model_columns[50:60]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
