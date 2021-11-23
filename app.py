from flask import Flask, request, jsonify

from detect.detect_image import DetectImage
from service.convert_image import ConvertImage

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/detect', methods=['GET'])
def foo():
    data = request.json
    # print(data['image'])
    image = data['image']
    response = DetectImage(image).detect()
    return jsonify(response)


if __name__ == '__main__':
    app.run()
