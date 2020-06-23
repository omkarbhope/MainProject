from flask import Flask, render_template, Response, request,redirect,url_for
from camera import VideoCamera


app = Flask(__name__)
_red = 0
_green = 0
_blue = 0
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/landing_page')
def landing_page():
    return render_template('landing_page.html')

@app.route('/categories')
def categories():
    return render_template('categories.html')


@app.route('/lipstick_categories')
def lipstick_categories():
    return render_template('lipstick_categories.html')


@app.route('/lip_products')
def lip_products():
    return render_template('lip_products.html')

@app.route('/matte_lipstick')
def matte_lipstick():
    return render_template('matte_lipstick.html')

@app.route('/soft_matte_cream_lipstick')
def soft_matte_cream_lipstick():
    return render_template('soft_matte_cream_lipstick.html')

@app.route('/lip_dip_lipstick')
def lip_dip_lipstick():
    return render_template('lip_dip_lipstick.html')

@app.route('/pure_matte_lipistick')
def pure_matte_lipistick():
    return render_template('pure_matte_lipistick.html')

@app.route('/lip_color_refill')
def lip_color_refill():
    return render_template('lip_color_refill.html')


@app.route('/triallist')
def triallist():
    return render_template('triallist.html')

@app.route('/tryout')
def tryout():
    return render_template('tryout.html')

@app.route('/product_description')
def product_description():
    return render_template('product_description.html')

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')


@app.route("/get_data", methods=["POST"])
def get_data():
    global _red,_green,_blue
    if request.method == "POST":
        red = int(request.form["red"])
        green = int(request.form["green"])
        blue = int(request.form["blue"])
        _red = red
        _blue = blue
        _green = green
        return render_template(url_for('tryout'))


def gen(camera):
    while True:
        frame = camera.get_frame(_red,_blue,_green)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080, debug=True)
