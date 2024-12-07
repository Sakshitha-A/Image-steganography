# Image Steganography and Watermarking Using Hybrid DCT-DWT

This project implements **image watermarking and steganography** using a hybrid approach of **Discrete Cosine Transform (DCT)** and **Discrete Wavelet Transform (DWT)**. The technique ensures robust and imperceptible watermark embedding and extraction.

---

## Features
- **Hybrid DCT-DWT Method** for robust frequency-domain watermarking.
- **Embedding and Extraction** of watermarks with minimal perceptual distortion.
- Support for common image formats: **JPG, PNG, BMP**.
- Customizable embedding strength for application-specific requirements.

---

## Requirements

### Dependencies
Ensure the following libraries are installed:

- **Python** (>= 3.7)
- **NumPy**
- **OpenCV**
- **PyWavelets**
- **Matplotlib**

Install dependencies using:
```bash
pip install numpy opencv-python pywavelets matplotlib
```

## Usage
Clone the Repository
```bash
git clone https://github.com/your-username/image-steganography-dct-dwt.git
cd image-steganography-dct-dwt
```

Input Files
Host Image: The image where the watermark will be embedded.
Watermark Image: The smaller image or logo to embed.
Embed the Watermark
Run the following command:

```bash
python embed.py --host host_image.jpg --watermark watermark_image.png --output watermarked_image.jpg
```
Arguments:

--host: Path to the host image.
--watermark: Path to the watermark image.
--output: Path to save the watermarked image.
Extract the Watermark
Run the following command:

```bash
python extract.py --input watermarked_image.jpg --output extracted_watermark.png
```

### Arguments

#### Embedding
- `--host`: Path to the host image.
- `--watermark`: Path to the watermark image.
- `--output`: Path to save the watermarked image.

#### Extraction
- `--input`: Path to the watermarked image.
- `--output`: Path to save the extracted watermark.

---

## How It Works

### Embedding Process
1. **Wavelet Decomposition**:  
   Decomposes the host image into subbands using **DWT**.
2. **Apply DCT**:  
   Applies **DCT** on selected subbands.
3. **Watermark Insertion**:  
   Embeds the watermark by modifying **DCT coefficients**.
4. **Reconstruction**:  
   Combines inverse **DCT** and **DWT** to produce the watermarked image.

### Extraction Process
1. **Wavelet Decomposition and DCT**:  
   Decomposes the watermarked image to locate the embedded watermark.
2. **Watermark Recovery**:  
   Extracts the watermark from **DCT coefficients**.
3. **Post-Processing**:  
   Restores the watermark to its original format.
