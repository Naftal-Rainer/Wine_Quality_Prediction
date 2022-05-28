from flask import Flask, render_template, request
import wine_Quality

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        result = request.form
    result = dict(result)  
    
    val = []
    for va in result.values():
        val.append(float(va))
    prediction = wine_Quality.model_data(val) 
    accuracy = wine_Quality.training_data_accuracy
    return render_template('result.html',result=prediction, acc = (round(accuracy,2)*100))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True)
