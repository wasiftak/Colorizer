import cv2
import numpy as np
import os

# ==== CONFIG SECTION ====
# Put your image path here (absolute or relative)
image_path = "Colorize/1.png"

# Model paths
prototxt_path = "models/colorization_deploy_v2.prototxt"
caffemodel_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
# ========================

# Get folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set results folder inside Colorize
output_folder = os.path.join(script_dir, "results")

# Create results folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Extract base name of image without extension
image_base_name = os.path.splitext(os.path.basename(image_path))[0]

# Build output filenames
colorized_filename = f"{image_base_name}_colorized.png"
comparison_filename = f"{image_base_name}_comparison.png"

colorized_path = os.path.join(output_folder, colorized_filename)
comparison_path = os.path.join(output_folder, comparison_filename)

# Load network and cluster centers
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
points = np.load(kernel_path)
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load image
bw_image = cv2.imread(image_path)
if bw_image is None:
    print("❌ Error: Cannot load image. Check path:", image_path)
    exit()

normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2Lab)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize ab to original size and combine
ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
L_original = cv2.split(lab)[0]
colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

# Save colorized image
cv2.imwrite(colorized_path, colorized)

# Create and save side-by-side comparison
comparison = np.hstack((bw_image, colorized))
cv2.imwrite(comparison_path, comparison)

# Show images
cv2.imshow("BW Image", bw_image)
cv2.imshow("Colorized Image", colorized)
cv2.imshow("Comparison", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Colorized image saved to: {colorized_path}")
print(f"✅ Comparison image saved to: {comparison_path}")
