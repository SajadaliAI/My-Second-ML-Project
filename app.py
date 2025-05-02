from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('Crop_Recommendation_App.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # This should match your HTML file name in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create feature list and reshape
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict using the loaded model
        prediction = model.predict(features)

        # Crop name dictionary (optional if model already returns string)
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        predicted_crop = crop_dict.get(prediction[0], str(prediction[0]))

        result = f"{predicted_crop} is the best crop to be cultivated right there."

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
