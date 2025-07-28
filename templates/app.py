from flask import Flask, request, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (replace 'your_model.pkl' with your model file)
try:
    model = joblib.load('rainfall_model.pkl')  # Change to your actual model file name
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please ensure 'rainfall_model.pkl' is in the correct directory.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        input_features = [float(x) for x in request.form.values()]
        
        # Ensure that input_features length matches the model's expected input shape
        if len(input_features) != model.n_features_in_:  # Check against the number of features in the model
            return render_template('index.html', prediction_text='Error: Incorrect number of input features.')

        input_data = np.array(input_features).reshape(1, -1)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'Predicted Annual Rainfall: {prediction[0]} mm')

    except ValueError as ve:
        return render_template('index.html', prediction_text=f'Error: {str(ve)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
