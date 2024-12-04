''' import os
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt

# Function to perform DCT (Discrete Cosine Transform)
def apply_dct(image):
    return cv2.dct(np.float32(image))

# Function to perform inverse DCT
def apply_idct(dct_image):
    return cv2.idct(dct_image)

# Function to apply DWT (Discrete Wavelet Transform)
def apply_dwt(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

# Function to perform inverse DWT
def apply_idwt(LL, LH, HL, HH):
    return pywt.idwt2((LL, (LH, HL, HH)), 'haar')

# Function to add watermark using Hybrid DCT and DWT
def add_watermark(base_image, watermark_image):
    # Check if watermark is smaller or equal to base image
    if base_image.shape[0] < watermark_image.shape[0] or base_image.shape[1] < watermark_image.shape[1]:
        print("Watermark is larger than the base image. Resizing watermark.")
        watermark_image = cv2.resize(watermark_image, (base_image.shape[1], base_image.shape[0]))

    # Split the base image into color channels
    b_channel, g_channel, r_channel = cv2.split(base_image)

    # Convert watermark to grayscale
    watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)

    # Function to watermark a single channel
    def watermark_channel(channel, watermark):
        # Apply DCT and DWT
        dct_channel = apply_dct(channel)
        LL, LH, HL, HH = apply_dwt(channel)

        # Resize watermark to fit the embedding region
        watermark_resized = cv2.resize(watermark, (LH.shape[1], LH.shape[0]))

        # Embed watermark (adjust strength for visibility)
        dct_channel[0:watermark_resized.shape[0], 0:watermark_resized.shape[1]] += watermark_resized * 0.05
        HL += watermark_resized * 0.05

        # Reconstruct the channel using inverse transforms
        watermarked_channel_dct = apply_idct(dct_channel)
        watermarked_channel_dwt = apply_idwt(LL, LH, HL, HH)

        # Combine DCT and DWT results (use DWT as base)
        watermarked_channel = np.clip(watermarked_channel_dwt, 0, 255).astype(np.uint8)
        return watermarked_channel

    # Watermark each channel
    b_watermarked = watermark_channel(b_channel, watermark_gray)
    g_watermarked = g_channel  # Keep untouched
    r_watermarked = r_channel  # Keep untouched

    # Resize channels to match the base image size if necessary
    b_watermarked = cv2.resize(b_watermarked, (base_image.shape[1], base_image.shape[0]))
    g_watermarked = cv2.resize(g_watermarked, (base_image.shape[1], base_image.shape[0]))
    r_watermarked = cv2.resize(r_watermarked, (base_image.shape[1], base_image.shape[0]))

    # Ensure all channels are of the same data type
    b_watermarked = b_watermarked.astype(np.uint8)
    g_watermarked = g_watermarked.astype(np.uint8)
    r_watermarked = r_watermarked.astype(np.uint8)

    # Merge watermarked channels back into a color image
    watermarked_image_color = cv2.merge([b_watermarked, g_watermarked, r_watermarked])

    return watermarked_image_color

#C:/Users/sakshita/OneDrive/Desktop/Mini Project/Code/text watermarknig/base.png
# Main program to load images and apply watermark
base_image_path = 'base.png'  # Replace with your base image path
watermark_image_path = 'base_image.png'  # Replace with your watermark image path

base_image = cv2.imread(base_image_path)
watermark_image = cv2.imread(watermark_image_path)

# Create a folder to save the images
output_folder = 'watermarked_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Apply watermark using Hybrid DCT-DWT
watermarked_image = add_watermark(base_image, watermark_image)

# Show the watermarked image
plt.imshow(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the base, watermark, and watermarked images to the folder
base_image_filename = os.path.join(output_folder, 'base_image.jpg')
watermark_image_filename = os.path.join(output_folder, 'watermark_image.jpg')
watermarked_image_filename = os.path.join(output_folder, 'watermarked_image.jpg')

cv2.imwrite(base_image_filename, base_image)
cv2.imwrite(watermark_image_filename, watermark_image)
cv2.imwrite(watermarked_image_filename, watermarked_image)

print(f"Images saved in {output_folder} folder.")
'''
import streamlit as st
import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from PIL import Image
import tempfile
import os

def embed_watermark_color(image, watermark, alpha=0.05):
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(yuv_img)

    coeffs = pywt.dwt2(y_channel, 'haar')
    LL, (LH, HL, HH) = coeffs

    watermark_resized = cv2.resize(watermark, (LL.shape[1], LL.shape[0]))

    dct_ll = dct(dct(LL.T, norm='ortho').T, norm='ortho')
    dct_ll[0:watermark_resized.shape[0], 0:watermark_resized.shape[1]] += alpha * watermark_resized
    ll_modified = idct(idct(dct_ll.T, norm='ortho').T, norm='ortho')

    watermarked_y = pywt.idwt2((ll_modified, (LH, HL, HH)), 'haar')
    watermarked_y = cv2.resize(watermarked_y, (y_channel.shape[1], y_channel.shape[0]))
    watermarked_y = np.clip(watermarked_y, 0, 255).astype(np.uint8)

    watermarked_yuv = cv2.merge((watermarked_y, u_channel, v_channel))
    watermarked_img = cv2.cvtColor(watermarked_yuv, cv2.COLOR_YUV2BGR)
    return watermarked_img

st.title("Steganography: Watermark Embedding")

# File uploads for base image and watermark
base_image = st.file_uploader("Upload Base Image", type=["jpg", "png", "jpeg"])
watermark_image = st.file_uploader("Upload Watermark Image (Grayscale)", type=["jpg", "png", "jpeg"])

# Slider for alpha value
alpha = st.slider("Adjust Embedding Intensity (Alpha)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

if st.button("Embed Watermark"):
    if base_image and watermark_image:
        # Convert uploaded files to OpenCV format
        base_image_cv = cv2.imdecode(np.frombuffer(base_image.read(), np.uint8), cv2.IMREAD_COLOR)
        watermark_image_cv = cv2.imdecode(np.frombuffer(watermark_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        try:
            # Embed the watermark
            watermarked_image = embed_watermark_color(base_image_cv, watermark_image_cv, alpha)
            
            # Convert to PIL for display
            watermarked_pil = Image.fromarray(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB))

            # Display the result
            st.image(watermarked_pil, caption="Watermarked Image", use_column_width=True)

            # Save to a temporary file for download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, watermarked_image)

            # Provide download button
            with open(temp_file.name, "rb") as file:
                st.download_button("Download Watermarked Image", data=file, file_name="watermarked_image.jpg", mime="image/jpeg")
            os.unlink(temp_file.name)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both the base image and the watermark.")