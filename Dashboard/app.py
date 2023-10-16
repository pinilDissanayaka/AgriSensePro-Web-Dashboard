from flask import Flask, request, render_template, url_for, Markup
import numpy as np
import pickle
import requests
import openai
import warnings
warnings.filterwarnings(action = 'ignore')

with open('utils\model\predictor.pickle', 'rb') as file:
    model = pickle.load(file)
    
def prediction(N, P, K, Ph, rf, city):
    pred = 0
    temp, hum = fetch_weather(city)
    input_data = [N, P, K, temp, hum, Ph, rf]
    input_data_arr = np.asarray(input_data).reshape(1, -1)
    pred = model.predict(input_data_arr)
    pred = pred[0]
    print(pred)
    return pred

def fetch_weather(city):
    api_key = '8afacb880aa75aed554fe64706531396'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temp = ((data['main']['temp']) - 273.15)
        humidity = data['main']['humidity']
        #desc = data['weather'][0]['description']
        #print(f'Temperature: {temp} C')
        #print(f"Humidity : {humidity}")
        return temp, humidity
    else:
        print('Error fetching weather data')
        
def GAN():
    openai.api_key = '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'
    prompt = "engineer."
    model = "text-davinci-003"
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=10)
    generated_text = response.choices[0].text
    print(generated_text)
    
    
app = Flask(__name__)

@app.route('/')
def dash():
    return render_template('dash.html')


@app.route('/crop_rec.html', methods=['POST', 'GET'])
def crop_rec(pred = 0):
    if request.method == 'POST':
        N = float(request.form['Nitrogen']) 
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        Ph = float(request.form['Ph'])
        R = float(request.form['Rainfall'])
        pred = prediction(N= N, P= P, K= K, Ph= Ph, rf= R, city= 'kolonnawa')
    return render_template('crop_rec.html', pred_value = pred)

@app.route('/plant_d.html')
def plant_d():
    return render_template('plant_d.html')

if __name__ == '__main__':
    app.run(debug=True)