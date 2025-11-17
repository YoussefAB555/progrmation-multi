import json
import sys
import os
from PIL import Image

# --- CONFIGURATION ---
# Target dimensions for the enlarged image (W x H)
W_new, H_new = 640, 480


def reconstruct_and_show_original(pixel_array):
    """
    Reconstructs the original image from the pixel array.

    Args:
        pixel_array (list): A list of lists, where each inner list is a row
                            of RGB pixel tuples/lists.

    Returns:
        tuple: (original_image, original_width, original_height)
    """
    original_height = len(pixel_array)
    if original_height == 0:
        raise ValueError("Pixel array is empty.")
    original_width = len(pixel_array[0])

    # 1. Create a new PIL Image object using 'RGB' mode
    original_image = Image.new('RGB', (original_width, original_height))

    # 2. Flatten the 2D pixel array into a 1D sequence of tuples for PIL's putdata
    flat_pixels = [tuple(pixel) for row in pixel_array for pixel in row]

    # 3. Load the data into the image object
    original_image.putdata(flat_pixels)

    print(f"Original image reconstructed. Size: {original_width}x{original_height}.")
    # To display the image, uncomment the line below. You may need to close the window
    # to allow the script to continue execution.
    # original_image.show(title="Original Image")

    return original_image, original_width, original_height


def enlarge_nearest_neighbor(original_image, pixel_array, original_width, original_height):
    """
    Strategy 1: Enlargement using Nearest Neighbor interpolation.
    This fulfills the requirement of 'scanning the pixels' manually.
    """
    enlarged_nn_image = Image.new('RGB', (W_new, H_new))
    enlarged_nn_pixels = []

    print(f"Strategy 1: Enlarging to {W_new}x{H_new} using Nearest Neighbor...")

    # Loop through every pixel in the *new*, enlarged image
    for y_new in range(H_new):
        # Calculate the corresponding row index in the *original* image
        # This determines which pixel to 'copy' to the current new position.
        y_orig = int(y_new * original_height / H_new)

        for x_new in range(W_new):
            # Calculate the corresponding column index in the *original* image
            x_orig = int(x_new * original_width / W_new)

            # Retrieve the pixel value and append it to the new, flattened list
            pixel_value = pixel_array[y_orig][x_orig]
            enlarged_nn_pixels.append(tuple(pixel_value))

    enlarged_nn_image.putdata(enlarged_nn_pixels)
    # enlarged_nn_image.show(title="Enlarged (Nearest Neighbor)")
    return enlarged_nn_image


def enlarge_smooth_interpolation_bonus(original_image):
    """
    Strategy 2 (Bonus): Enlargement using Bilinear Interpolation.
    This offers a smoother result than Nearest Neighbor by averaging adjacent pixels.
    We use the PIL library's implementation for simplicity and performance.
    """
    print(f"Strategy 2 (Bonus): Enlarging to {W_new}x{H_new} using Bilinear Interpolation...")

    # Image.Resampling.BILINEAR performs the gradient interpolation required.
    enlarged_smooth_image = original_image.resize((W_new, H_new), Image.Resampling.BILINEAR)

    # enlarged_smooth_image.show(title="Enlarged (Bilinear/Smooth Bonus)")
    return enlarged_smooth_image


def apply_blur_convolution(source_image):
    """
    Applies a 3x3 averaging blur convolution filter (Kernel).

    The convolution operation involves iterating through each pixel (x, y)
    in the source image and applying a weighted sum of its neighbors based on the kernel.
    """
    # 3x3 Averaging Blur Kernel (or Matrix)
    kernel = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    kernel_sum = sum(sum(row) for row in kernel)  # Sum of coefficients = 9
    K_size = len(kernel)
    K_half = K_size // 2  # = 1 (radius)

    blurred_image = Image.new('RGB', (W_new, H_new))
    # Load pixels for fast read/write access
    source_pixels = source_image.load()
    dest_pixels = blurred_image.load()

    print("Applying 3x3 Blur Convolution Filter...")

    # Iterate through every pixel in the new image
    for y in range(H_new):
        for x in range(W_new):
            R_total, G_total, B_total = 0, 0, 0

            # Perform the convolution product: loop over the kernel's neighbors
            for ky in range(K_size):
                for kx in range(K_size):
                    # Calculate the neighbor's coordinate in the source image
                    source_x = x + kx - K_half
                    source_y = y + ky - K_half

                    # Boundary check: Skip or clamp pixels outside the image bounds
                    if 0 <= source_x < W_new and 0 <= source_y < H_new:
                        R, G, B = source_pixels[source_x, source_y]
                        weight = kernel[ky][kx]

                        # Accumulate the weighted sum for each color channel
                        R_total += R * weight
                        G_total += G * weight
                        B_total += B * weight

            # Final step: Divide the result by the sum of the kernel coefficients
            R_new = int(R_total / kernel_sum)
            G_new = int(G_total / kernel_sum)
            B_new = int(B_total / kernel_sum)

            # Ensure color values are within the valid 0-255 range
            dest_pixels[x, y] = (
                max(0, min(255, R_new)),
                max(0, min(255, G_new)),
                max(0, min(255, B_new))
            )

    # blurred_image.show(title="Blurred Image (Convolution)")
    return blurred_image


def apply_jpeg_compression(blurred_image):
    """
    Saves the blurred image in JPEG format at different quality levels
    and reports the file size for comparison.
    """
    qualities = [25, 50, 95]
    print("\n--- JPEG Compression Results (Quality vs. File Size) ---")
    print("| Quality | File Size (KB) |")
    print("|---------|----------------|")

    for quality in qualities:
        filename = f"blurred_q{quality}.jpg"
        # PIL's 'quality' parameter controls the compression strength (0-100)
        blurred_image.save(filename, 'JPEG', quality=quality)

        file_size_kb = os.path.getsize(filename) / 1024
        print(f"| {quality:^7} | {file_size_kb:>14.2f} |")

    print("\nComparison Notes:")
    print(" - **Quality 95**: Highest quality, largest file size. Minimal visual artifacts.")
    print(" - **Quality 50**: Good compromise between visual quality and file size.")
    print(" - **Quality 25**: Lowest quality, smallest file size. **High artifact visibility** (blockiness/fuzziness).")


if __name__ == "__main__":
    # --- 1. Load Data ---
    if len(sys.argv) != 2:
        print("Usage: python 'pythonfilename.py' imagemystere.json")
        sys.exit(1)

    json_filename = sys.argv[1]

    try:
        # Open and load the JSON file
        with open(json_filename, 'r') as f:
            data = json.load(f)
            # The actual pixel array is nested under the 'pixels' key
            pixel_array = data["pixels"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading JSON data: {e}")
        sys.exit(1)

    # --- 2. Reconstruct and Display Original ---
    original_img, W_orig, H_orig = reconstruct_and_show_original(pixel_array)

    # --- 3. Image Transformation (Enlargement) ---
    # Strategy 1: Required manual pixel scanning implementation
    enlarged_nn = enlarge_nearest_neighbor(original_img, pixel_array, W_orig, H_orig)

    # Strategy 2: Bonus Implementation (Using PIL's optimized interpolation)
    enlarged_smooth = enlarge_smooth_interpolation_bonus(original_img)

    # --- 4. Filter and Convolution Matrix (Blur) ---
    # We apply the convolution filter to the Nearest Neighbor enlarged image
    blurred_img = apply_blur_convolution(enlarged_nn)

    # --- 5. Apply JPEG Compression ---
    apply_jpeg_compression(blurred_img)

    print("\n--- Script finished. Check the current directory for the generated .jpg files. ---")