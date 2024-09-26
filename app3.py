from flask import Flask, render_template, request, jsonify
import google.generativeai as palm

# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

# Configure the AI API key
palm_api_key = "AIzaSyBp1wcksd55hbKK2lJtvBoNFWiaWND3QMU"
palm.configure(api_key=palm_api_key)

@app.route('/')
def index():
    return render_template('New.html')  # Render the new.html template

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = palm.chat(messages=user_input)
    truncated_response = response.last[:100] if len(response.last) > 100 else response.last
    return jsonify({'response': truncated_response})

if __name__ == '__main__':
    app.run(debug=True)
