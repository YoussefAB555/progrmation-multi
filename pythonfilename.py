import json
import sys
import os
from PIL import Image

# --- CONFIGURATION ---
W_new, H_new = 640, 480 

def reconstruct_and_show_original(pixel_array):
    """Reconstructs the original image from the pixel array."""
    original_height = len(pixel_array)
    if original_height == 0:
        raise ValueError("Pixel array is empty.")
    original_width = len(pixel_array[0])
    
    original_image = Image.new('RGB', (original_width, original_height))
    flat_pixels = [tuple(pixel) for row in pixel_array for pixel in row]
    original_image.putdata(flat_pixels)
    
    print(f"Original image reconstructed. Size: {original_width}x{original_height}.")
    # original_image.show(title="Original Image") 
    
    return original_image, original_width, original_height

def enlarge_nearest_neighbor(original_image, pixel_array, original_width, original_height):
    """Strategy 1: Enlargement using Nearest Neighbor interpolation."""
    enlarged_nn_image = Image.new('RGB', (W_new, H_new))
    enlarged_nn_pixels = []

    print(f"Strategy 1: Enlarging to {W_new}x{H_new} using Nearest Neighbor...")
    
    for y_new in range(H_new):
        y_orig = int(y_new * original_height / H_new)
        
        for x_new in range(W_new):
            x_orig = int(x_new * original_width / W_new)
            
            pixel_value = pixel_array[y_orig][x_orig]
            enlarged_nn_pixels.append(tuple(pixel_value))

    enlarged_nn_image.putdata(enlarged_nn_pixels)
    # enlarged_nn_image.show(title="Enlarged (Nearest Neighbor)") 
    return enlarged_nn_image

def enlarge_smooth_interpolation_bonus(original_image):
    """Strategy 2 (Bonus): Enlargement using Bilinear Interpolation (PIL method)."""
    print(f"Strategy 2 (Bonus): Enlarging to {W_new}x{H_new} using Bilinear Interpolation...")
    enlarged_smooth_image = original_image.resize((W_new, H_new), Image.Resampling.BILINEAR)
    # enlarged_smooth_image.show(title="Enlarged (Bilinear/Smooth Bonus)") 
    return enlarged_smooth_image

def apply_blur_convolution(source_image):
    """Applies a 3x3 averaging blur convolution filter (Kernel)."""
    kernel = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    kernel_sum = sum(sum(row) for row in kernel)
    K_size = len(kernel)
    K_half = K_size // 2  

    blurred_image = Image.new('RGB', (W_new, H_new))
    source_pixels = source_image.load()
    dest_pixels = blurred_image.load()
    
    print("Applying 3x3 Blur Convolution Filter...")

    for y in range(H_new):
        for x in range(W_new):
            R_total, G_total, B_total = 0, 0, 0
            
            for ky in range(K_size):
                for kx in range(K_size):
                    source_x = x + kx - K_half
                    source_y = y + ky - K_half
                    
                    if 0 <= source_x < W_new and 0 <= source_y < H_new:
                        R, G, B = source_pixels[source_x, source_y]
                        weight = kernel[ky][kx]
                        
                        R_total += R * weight
                        G_total += G * weight
                        B_total += B * weight

            R_new = int(R_total / kernel_sum)
            G_new = int(G_total / kernel_sum)
            B_new = int(B_total / kernel_sum)
            
            dest_pixels[x, y] = (
                max(0, min(255, R_new)),
                max(0, min(255, G_new)),
                max(0, min(255, B_new))
            )

    # blurred_image.show(title="Blurred Image (Convolution)") 
    return blurred_image

def apply_jpeg_compression(blurred_image):
    """Saves the image in JPEG format with different quality levels."""
    qualities = [25, 50, 95]
    print("\n--- JPEG Compression Results (Quality vs. File Size) ---")
    print("| Quality | File Size (KB) |")
    print("|---------|----------------|")

    for quality in qualities:
        filename = f"blurred_q{quality}.jpg"
        blurred_image.save(filename, 'JPEG', quality=quality)
        
        file_size_kb = os.path.getsize(filename) / 1024
        print(f"| {quality:^7} | {file_size_kb:>14.2f} |")

    print("\nComparison Notes: Quality 95 has best visual quality/largest size; Quality 25 has lowest visual quality/smallest size.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 'pythonfilename.py' imagemystere.json")
        sys.exit(1)

    json_filename = sys.argv[1]

    try:
        with open(json_filename, 'r') as f:
            data = json.load(f)
            pixel_array = data["pixels"] 
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        sys.exit(1)

    original_img, W_orig, H_orig = reconstruct_and_show_original(pixel_array)
    enlarged_nn = enlarge_nearest_neighbor(original_img, pixel_array, W_orig, H_orig)
    enlarged_smooth = enlarge_smooth_interpolation_bonus(original_img)
    blurred_img = apply_blur_convolution(enlarged_nn)
    apply_jpeg_compression(blurred_img)
    
    print("\n--- Script finished. Check the current directory for the generated .jpg files. ---")

