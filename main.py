import os
import gdown
import numpy as np
import cv2

def download_models():
    os.makedirs("models", exist_ok=True)

    caffemodel_url = "https://drive.google.com/uc?id=1EshSEknFNC0eknpyLk39N1x-PKRi4Vpv" 
    prototxt_url =    "https://drive.google.com/uc?id=1hzb13B8UUxcc8hgGzukZNZHwqEDtItMY"
    hull_url = "https://drive.google.com/uc?id=1F1Prs7yDhoQwIJ36Qa3a1dAqtd6F5UhT"

    if not os.path.exists("models/colorization_release_v2.caffemodel"):
        gdown.download(caffemodel_url, "models/colorization_release_v2.caffemodel", quiet=False)

    if not os.path.exists("models/colorization_deploy_v2.prototxt"):
        gdown.download(prototxt_url, "models/colorization_deploy_v2.prototxt", quiet=False)

    if not os.path.exists("models/pts_in_hull.npy"):
        gdown.download(hull_url, "models/pts_in_hull.npy", quiet=False)

# Download models if not present
download_models()

def colorize_image(input_path, output_path):
    prototxt_path = "models/colorization_deploy_v2.prototxt"
    caffemodel_path = "models/colorization_release_v2.caffemodel"
    kernel_path = "models/pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    bw_image = cv2.imread(input_path)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2Lab)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
    colorized = (255 * colorized).astype("uint8")

    cv2.imwrite(output_path, colorized)
