# 🖌️ Colorize: Black & White Image Colorization

This project colorizes black and white images using a pretrained deep learning model (Caffe) with OpenCV.

## 💡 How It Works

We use a pretrained CNN model (`colorization_release_v2.caffemodel`) to predict color components (AB channels) for grayscale images.  
These predictions are combined with the original lightness channel (L) in the LAB color space to generate realistic colorized images.

## 🚀 How to Run (Local)

1. 🔧 **Install the required libraries**:

   ```bash
   pip install -r requirements.txt
   ```

2. 🧠 **No need to manually download model files!**  
   On first run, the required model files will be downloaded automatically via `gdown`.

3. ▶️ **Start the Flask server**:

   ```bash
   python app.py
   ```

4. 🌐 Open your browser and go to:  
   [http://localhost:10000](http://localhost:10000)

5. 📸 Upload a black-and-white image — get back a colorized result instantly.
