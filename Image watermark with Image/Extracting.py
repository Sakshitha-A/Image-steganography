import cv2
import numpy as np
from scipy.fftpack import dct, idct
import pywt
import os

def extract_watermark_grayscale(image, alpha=0.05):
    """
    Extracts watermark from a watermarked grayscale image using DWT and DCT.
    
    Parameters:
    - image: Path to the watermarked image.
    - alpha: Scaling factor for watermark strength.
    
    Returns:
    - extracted_watermark: Extracted watermark in grayscale.
    """
    # Read the watermarked image in grayscale
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error loading image. Ensure file path is correct.")
    
    # Get the size of the image (height, width)
    watermark_size = img.shape

    # Apply DWT to extract the LL sub-band
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Apply DCT to the LL sub-band
    dct_ll = dct(dct(LL.T, norm='ortho').T, norm='ortho')

    # Extract watermark from the DCT coefficients
    extracted_watermark = dct_ll[:watermark_size[0], :watermark_size[1]] / alpha

    # Clip and normalize the extracted watermark
    extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)

    # Apply denoising or blurring to enhance the image
    extracted_watermark = cv2.GaussianBlur(extracted_watermark, (5, 5), 0)

    return extracted_watermark

# Example usage
try:
    # Create a folder to save extracted watermarks
    output_folder = 'extracted_watermarks'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set alpha value to 0.05 as requested
    alpha = 0.2

    # Path to the watermarked image
    watermarked_image_path = 'C:\\Users\\sakshita\\OneDrive\\Desktop\\Mini Project\\Code\\water4.jpg'  # Replace with your image path

    # Extract watermark with alpha = 0.05 and dynamic image size
    extracted_watermark = extract_watermark_grayscale(watermarked_image_path, alpha=alpha)
    
    # Save the extracted watermark
    output_filename = os.path.join(output_folder, f'extracted_watermark_alpha_{alpha:.2f}51.jpg')
    cv2.imwrite(output_filename, extracted_watermark)
    print(f"Extracted watermark saved as '{output_filename}'")

except ValueError as e:
    print(e)
