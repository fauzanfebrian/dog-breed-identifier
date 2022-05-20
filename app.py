import base64
import os
from utilities import predict
from datetime import datetime
from flask import Flask, render_template, request, jsonify


app = Flask(__name__,
            static_url_path='', 
            static_folder='static')

@app.route('/')
def home():
   return render_template('home.html')

@app.post("/identify")
def post_identify():
   content = request.json
   if "image" not in content:
      return jsonify({"error": "Image should be filled"})
   filename = convert_and_save(content["image"])
   preds = predict([filename])
   os.unlink(filename)
   return jsonify({"preds": preds})
   
   
def convert_and_save(b64_string):
   b64_string += '=' * (-len(b64_string) % 4)  # restore stripped '='s
   string = b'{b64_string}'
   with open(f"images/{datetime.now()}.jpeg", "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))
        filename = fh.name
   return filename
   
if __name__ == '__main__':
   app.run()
