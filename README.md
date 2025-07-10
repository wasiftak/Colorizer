# Colorize: Black & White Image Colorization

This project colorizes black and white images using a pretrained Caffe model with OpenCV.

## 💡 How it works

We use a pretrained CNN model (colorization_release_v2.caffemodel) to predict color (AB channels) from the grayscale (L channel) of LAB color space.

## 🏃 How to run

1. Download the pretrained models from [colorization models link](https://drive.google.com/drive/folders/1FaDajjtAsntF_Sw5gqF0WyakviA5l8-a).
2. Place them in a folder named `models/`.
3. Put your black-and-white images in the project root (e.g., `1.png`).
4. Install requirements:

```bash
pip install -r requirements.txt
