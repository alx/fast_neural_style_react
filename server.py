#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import subprocess
import glob
import base64

neural_style_script= "./fast_neural_style/neural_style/neural_style.py"
style_dir_path= "./source/"
checkpoints_dir_path= "./checkpoints/"

screenshot_image = "./screenshot.jpg"
output_image = "./latest.jpg"

base_path = os.path.dirname(__file__)

class StaticServer(SimpleHTTPRequestHandler):

    def do_POST(self):
       body = json.loads(request.content.read())
       with open(screenshot_image, "wb") as fh:
           fh.write(base64.decodebytes(body.screenshot))
       subprocess.call([
           "python",
           neural_style_script,
           "eval",
           "--content-image",
           screenshot_image,
           "--model",
           body.modelPath,
           "--output-image",
           output_image,
           "--cuda",
           "1"
       ])

    def do_GET(self):
        f = self.send_head()
        if f:
            try:
                self.copyfile(f, self.wfile)
            finally:
                f.close()

def run(server_class=HTTPServer, handler_class=StaticServer, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting Server on port {}'.format(port))
    httpd.serve_forever()

run()
