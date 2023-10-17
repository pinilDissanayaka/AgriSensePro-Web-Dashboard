from flask import Flask, request, render_template, url_for, Markup
import numpy as np
import pickle
import os
import replicate
import requests
#import openai
import warnings
warnings.filterwarnings(action = 'ignore')

api_key = '8afacb880aa75aed554fe64706531396' #Weather api
os.environ["REPLICATE_API_TOKEN"] = "r8_3GNMwe3ZfwbZAu7OGb0WTxZPR2SzYbD061RJy" #Llama2 Api


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
        
#def GAN():
#    openai.api_key = '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'
#    prompt = "engineer."
#    model = "text-davinci-003"
#    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=10)
#    generated_text = response.choices[0].text
#    print(generated_text)

def llama2(prompt):
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                        input={"prompt": f"{prompt}", # Prompts
                        "temperature":0.1, "top_p":0.9, "max_length":2048, "repetition_penalty":1})  # Model parameters
    full_response = ""
    for item in output:
        full_response += item

    print(full_response)
    
    
    
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