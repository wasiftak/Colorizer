import numpy as np
import cv2
import os

# Set model paths
prototxt_path = "models/colorization_deploy_v2.prototxt"
caffemodel_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"


# Load model only once
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]


def colorize_image(image_path, output_path):
    # Load the B&W image
    bw_image = cv2.imread(image_path)
    if bw_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2Lab)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L_original = cv2.split(lab)[0]

    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
    colorized = (255 * colorized).astype("uint8")

    # Save result
    cv2.imwrite(output_path, colorized)
    print(f"Colorized image saved to {output_path}")

