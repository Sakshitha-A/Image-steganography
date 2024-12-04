import cv2
import numpy as np
import pywt
import os

# Function to perform DWT (Discrete Wavelet Transform)
def apply_dwt(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

# Function to perform inverse DWT
def apply_idwt(LL, LH, HL, HH):
    return pywt.idwt2((LL, (LH, HL, HH)), 'haar')

# Function to embed text in the image using Hybrid DWT
def embed_text(image, secret_text):
    # Convert secret text to binary
    binary_text = ''.join(format(ord(char), '08b') for char in secret_text)

    # Split the image into color channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Function to embed text in a single channel
    def embed_channel(channel, binary_text):
        # Apply DWT on the channel
        LL, LH, HL, HH = apply_dwt(channel)

        # Cast LL to an integer type for bitwise operations
        LL = LL.astype(np.int32)

        # Embed the binary text into the LL (Low-Low) sub-band of the DWT coefficients
        binary_index = 0
        for i in range(LL.shape[0]):
            for j in range(LL.shape[1]):
                if binary_index < len(binary_text):
                    # Change the LSB (Least Significant Bit) of the DWT coefficients
                    LL[i, j] = (LL[i, j] & ~1) | int(binary_text[binary_index])
                    binary_index += 1

        # Convert LL back to float32 after embedding
        LL = LL.astype(np.float32)

        # Reconstruct the image using inverse DWT with the modified LL coefficients
        watermarked_channel = apply_idwt(LL, LH, HL, HH)

        return np.clip(watermarked_channel, 0, 255).astype(np.uint8)

    # Embed the text in each color channel
    b_watermarked = embed_channel(b_channel, binary_text)
    g_watermarked = embed_channel(g_channel, binary_text)
    r_watermarked = embed_channel(r_channel, binary_text)

    # Merge the watermarked channels back into a color image
    watermarked_image = cv2.merge([b_watermarked, g_watermarked, r_watermarked])

    return watermarked_image

# Function to save the image
def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Watermarked image saved at: {output_path}")

# Main program
def main_embed():
    # Load the image where the text will be embedded
    image_path = 'C:/Users/sakshita/OneDrive/Desktop/Mini Project/Code/text watermarknig/base.png'  # Replace with the path to your image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return

    # Text to embed
    secret_text = "This is a secret watermark"

    # Embed the text in the image
    watermarked_image = embed_text(image, secret_text)

    # Show the watermarked image
    cv2.imshow("Watermarked Image", watermarked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the watermarked image
    output_folder = 'C:/Users/sakshita/OneDrive/Desktop/Mini Project/Code/text watermarknig'  # Specify the folder
    if not os.path.exists(output_folder):  # Create the folder if it doesn't exist
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'watermarked_image.png')  # Specify the file name within the folder
    save_image(watermarked_image, output_path)


# Run the program
if __name__ == "__main__":
    main_embed()
