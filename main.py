import numpy as np
import cv2
import os

def colorize_image(image_path, output_path):
    """
    This function takes a path to a grayscale image, colorizes it using a
    pre-trained Caffe model, and saves the result.
    """
    try:
        # --- ROBUST PATH CONSTRUCTION ---
        # Get the absolute path to the directory where this script is located.
        # This is the key to making the script work reliably.
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the paths to the model files relative to the script's directory.
        prototxt_path = os.path.join(script_dir, 'models', 'colorization_deploy_v2.prototxt')
        caffemodel_path = os.path.join(script_dir, 'models', 'colorization_release_v2.caffemodel')
        points_path = os.path.join(script_dir, 'models', 'pts_in_hull.npy')

        # --- VERIFICATION (Important for Debugging) ---
        # Print the absolute paths to ensure they are correct.
        # Before running, manually check if these files exist at these locations.
        print("--- Loading Model Files ---")
        print(f"Prototxt:   {prototxt_path}")
        print(f"CaffeModel: {caffemodel_path}")
        print(f"NumPy Points: {points_path}")
        print("---------------------------\n")

        # --- MODEL LOADING ---
        # Load the Caffe model from disk.
        print("[INFO] Loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # Load the cluster center points (the "kernel").
        pts = np.load(points_path)

        # Add the cluster centers as 1x1 convolutions to the model.
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs.append(pts)
        net.getLayer(conv8).blobs.append(np.full([1, 313], 2.606, dtype="float32"))

        # --- IMAGE PROCESSING ---
        # Load the input image, scale it, and convert to LAB color space.
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not read image from path: {image_path}")
            return

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Resize the L channel to the dimensions the model expects.
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50  # Mean-centering

        # --- PREDICTION ---
        # Pass the L channel through the network to predict the a and b channels.
        print("[INFO] Colorizing image...")
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        # Resize the predicted ab channels to match the original image dimensions.
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # --- RECONSTRUCTION ---
        # Concatenate the original L channel with the predicted ab channels.
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # Convert the LAB image back to BGR color space.
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)

        # Convert back to 8-bit integer range.
        colorized = (255 * colorized).astype("uint8")

        # --- SAVE RESULT ---
        cv2.imwrite(output_path, colorized)
        print(f"[SUCCESS] Colorized image saved to: {output_path}")

        # Optional: Display the images
        # cv2.imshow("Original", image)
        # cv2.imshow("Colorized", colorized)
        # cv2.waitKey(0)

    except cv2.error as e:
        print("[ERROR] OpenCV error. This often means a model file is missing or corrupt.")
        print(f"Details: {e}")
    except FileNotFoundError:
        print("[ERROR] A file was not found. Please check your file paths and ensure the 'models' directory is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- USAGE EXAMPLE ---
    # Make sure you have an 'images' folder with your test image
    # and an 'outputs' folder for the results.
    input_image_path = 'images/your_test_image.jpg' # CHANGE THIS
    output_image_path = 'outputs/colorized_result.jpg' # CHANGE THIS

    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    colorize_image(input_image_path, output_image_path)
