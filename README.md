# fast_neural_style_react

Project to capture image from a webcam in a browser, send it to pytorch fast_neural_style style transfer example, and then display its result in a html canvas.

Models can be changed from the browser.

* React Webcam: https://github.com/mozmorris/react-webcam
* PyTorch Fast Neural Style example: https://github.com/pytorch/examples/tree/master/fast_neural_style

## Usage

* Download models

```
python download_saved_models.py
```

* Start python server


```
python server.py
# open browser on http://localhost:8000
```
