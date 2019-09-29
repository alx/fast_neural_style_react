#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import subprocess
import glob
import base64
import json
from io import BytesIO
import io
import re
from PIL import Image

import torch
from torchvision import transforms
from transformer_net import TransformerNet

neural_style_script= "./fast_neural_style/neural_style/neural_style.py"
style_dir_path= "./source/"
checkpoints_dir_path= "./checkpoints/"

screenshot_image = "./screenshot.jpg"
output_image = "./latest.jpg"

base_path = os.path.dirname(__file__)

currentModel = './saved_models/candy.pth'
device = torch.device("cuda")
style_model = TransformerNet()

def loadStyleModel(path, firstLoad):

  if firstLoad == False and path == currentModel:
      print("same model")
      return

  with torch.no_grad():
      state_dict = torch.load(path)
      # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
      for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
          del state_dict[k]
      style_model.load_state_dict(state_dict)
      style_model.to(device)

loadStyleModel(currentModel, True)

class StaticServer(SimpleHTTPRequestHandler):


    def do_POST(self):
      print("do_POST")
      self.data_string = self.rfile.read(int(self.headers['Content-Length']))
      body = json.loads(self.data_string.decode('utf-8'))

      loadStyleModel(body['model'], False)

#       with open(screenshot_image, "wb") as fh:
#           fh.write(base64.b64decode(body['screenshot']))
#       self.send_response(200)
#       self.end_headers()

      msg = base64.b64decode(body['screenshot'])
      buf = io.BytesIO(msg)
      content_image = Image.open(buf)

      content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
      content_image = content_transform(content_image)
      content_image = content_image.unsqueeze(0).to(device)

      with torch.no_grad():
          output = style_model(content_image).cpu()

      img = output[0].clone().clamp(0, 255).numpy()
      img = img.transpose(1, 2, 0).astype("uint8")
      img = Image.fromarray(img)

      buffered = BytesIO()
      img.save(buffered, format="JPEG")
      img_str = base64.b64encode(buffered.getvalue())

      self.protocol_version = 'HTTP/1.1'
      self.send_response(200, 'OK')
      self.end_headers()
      self.wfile.write(img_str)
      # self.path = '/'
      return

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
