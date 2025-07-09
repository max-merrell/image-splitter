import os
from PIL import Image
import sys
import cv2
import numpy as np

def find_vertical_black_line_center(image_path, black_threshold=25, min_line_width=5, min_line_density=0.95, search_middle_fraction=0.2):
    """
    Attempts to find the center of a prominent vertical black line in an image,
    restricting the search to the middle fifth of the photo's width.

    Args:
        image_path (str): Path to the image file.
        black_threshold (int): Pixels with intensity below this are considered "black" (0-255).
        min_line_width (int): Minimum width (in pixels) for a continuous black region to be considered a line.
        min_line_density (float): Minimum proportion of black pixels in a column to be considered part of the line (0.0 to 1.0).
        search_middle_fraction (float): The fraction of the image width to search in the middle (e.g., 0.2 for middle fifth).

    Returns:
        int or None: The x-coordinate of the center of the detected black line, or None if no line is found.
    """
    try:
        # Load image with OpenCV
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            print(f"Warning: Could not read image {image_path} with OpenCV.")
            return None

        height, width = img_cv.shape

        # Define the search region for the line
        search_width = int(width * search_middle_fraction)
        search_start_x = (width - search_width) // 2
        search_end_x = search_start_x + search_width

        # Ensure bounds are within image dimensions
        search_start_x = max(0, search_start_x)
        search_end_x = min(width, search_end_x)

        # Apply a binary threshold to isolate black pixels
        # Pixels <= black_threshold become 0 (black), others become 255 (white)
        _, binary_img = cv2.threshold(img_cv, black_threshold, 255, cv2.THRESH_BINARY)
        
        # Sum black pixels vertically (column-wise) specifically in the search region
        column_sums = np.sum(binary_img == 0, axis=0) # Sum 0s (black pixels)

        # Normalize column sums by height to get density
        column_densities = column_sums / height

        # Find potential line regions within the search_start_x and search_end_x
        potential_line_centers = []
        is_in_line = False
        current_line_start = -1

        for x in range(search_start_x, search_end_x):
            if column_densities[x] >= min_line_density:
                if not is_in_line:
                    current_line_start = x
                    is_in_line = True
            else:
                if is_in_line:
                    current_line_end = x - 1
                    line_width = current_line_end - current_line_start + 1
                    if line_width >= min_line_width:
                        potential_line_centers.append((current_line_start + current_line_end) // 2)
                    is_in_line = False
        
        # Check if a line extends to the end of the search region
        if is_in_line:
            current_line_end = search_end_x - 1
            line_width = current_line_end - current_line_start + 1
            if line_width >= min_line_width:
                potential_line_centers.append((current_line_start + current_line_end) // 2)

        if not potential_line_centers:
            # print(f"No clear black line found in the middle fifth of {os.path.basename(image_path)}.")
            return None
        
        # If multiple lines are found, take the one that's densest or widest.
        # For simplicity, let's take the first one found in the search area for now,
        # assuming a single prominent line. If a more sophisticated choice is needed,
        # we'd iterate through potential_line_starts/ends and evaluate them.
        
        # For now, let's just return the center of the first valid line found.
        # If there are multiple, the one closer to the exact middle of the *search region* might be best.
        # Let's refine to pick the "best" line, e.g., the widest one in the search area
        best_line_center = None
        max_line_width_found = 0

        # Re-iterate to find the widest line within the search region
        is_in_line = False
        current_line_start = -1
        for x in range(search_start_x, search_end_x):
            if column_densities[x] >= min_line_density:
                if not is_in_line:
                    current_line_start = x
                    is_in_line = True
            else:
                if is_in_line:
                    current_line_end = x - 1
                    line_width = current_line_end - current_line_start + 1
                    if line_width >= min_line_width:
                        if line_width > max_line_width_found:
                            max_line_width_found = line_width
                            best_line_center = (current_line_start + current_line_end) // 2
                    is_in_line = False
        
        # Final check if line extends to the end of search region
        if is_in_line:
            current_line_end = search_end_x - 1
            line_width = current_line_end - current_line_start + 1
            if line_width >= min_line_width:
                if line_width > max_line_width_found:
                    max_line_width_found = line_width
                    best_line_center = (current_line_start + current_line_end) // 2

        return best_line_center

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)} for line detection: {e}")
        return None


def split_jpeg_in_half_by_line(folder_path):
    """
    Splits each JPEG image in a given folder vertically based on a detected black line
    in the middle fifth of the photo. If no line is found in that region,
    it falls back to splitting exactly in half.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        input("Press Enter to exit.")
        return

    output_folder = os.path.join(folder_path, "split_images_by_line")
    os.makedirs(output_folder, exist_ok=True)

    processed_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size

                    # Pass the search_middle_fraction parameter here
                    split_x = find_vertical_black_line_center(file_path, search_middle_fraction=0.2) # 0.2 for middle fifth
                    
                    if split_x is None:
                        # Fallback to exact center if no line detected in the middle fifth
                        split_x = width // 2
                        print(f"No clear line found in the middle fifth of '{filename}'. Splitting at geometric center ({split_x}).")
                    else:
                        print(f"Detected line in '{filename}' at x-coordinate: {split_x}. Splitting there.")


                    # Crop the left half
                    # Ensure the split_x doesn't cause negative width for left_half or zero width for right_half
                    split_x = max(1, min(split_x, width - 1)) # Ensure split_x is within valid bounds (not 0 or width)

                    left_half = img.crop((0, 0, split_x, height))
                    right_half = img.crop((split_x, 0, width, height))

                    base_name, ext = os.path.splitext(filename)

                    left_half_name = f"{base_name}(2){ext}"
                    right_half_name = f"{base_name}(1){ext}"

                    left_half.save(os.path.join(output_folder, left_half_name))
                    right_half.save(os.path.join(output_folder, right_half_name))

                    print(f"Saved '{left_half_name}' and '{right_half_name}'.")
                    processed_count += 1

            except Exception as e:
                print(f"Could not process '{filename}': {e}")
    
    if processed_count > 0:
        print(f"\nFinished splitting {processed_count} images. All split images are saved in the '{output_folder}' folder.")
    else:
        print("\nNo JPEG images found in this folder to split.")
    
    input("Press Enter to exit.")

# --- How to use the script ---
if __name__ == "__main__":
    jpeg_folder = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    print(f"Looking for JPEGs in: {jpeg_folder}")
    print("Splitting images now...")
    split_jpeg_in_half_by_line(jpeg_folder)