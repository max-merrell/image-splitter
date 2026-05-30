import os
from PIL import Image
import sys
import cv2
import numpy as np

def find_vertical_black_line_center(image_path, black_threshold=25, min_line_width=5, min_line_density=0.9, search_middle_fraction=0.2):
    """
    Attempts to find the center of a prominent vertical black line in an image,
    restricting the search to the middle fifth of the photo's width.
    """
    try:
        # Load image with OpenCV
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            return None

        height, width = img_cv.shape

        # Define the search region for the line (middle fifth)
        search_width = int(width * search_middle_fraction)
        search_start_x = (width - search_width) // 2
        search_end_x = search_start_x + search_width

        # Ensure bounds are within image dimensions
        search_start_x = max(0, search_start_x)
        search_end_x = min(width, search_end_x)

        # Apply a binary threshold to isolate black pixels
        _, binary_img = cv2.threshold(img_cv, black_threshold, 255, cv2.THRESH_BINARY)
        
        # Sum black pixels vertically across the ENTIRE image width to match index sizing
        column_sums = np.sum(binary_img == 0, axis=0) 

        # Normalize column sums by height to get density
        column_densities = column_sums / height

        # Find potential line regions within the search bounds
        best_line_center = None
        max_line_width_found = 0

        is_in_line = False
        current_line_start = -1
        
        # Iterate only within the defined search region
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
        
        # Final check if a line extends to the end of the search region
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


def split_jpeg_in_half_by_line(folder_path, flip_180=False):
    """
    Splits each JPEG image in a given folder vertically based on a detected black line.
    Optionally rotates the split output images 180 degrees.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' was not found.")
        input("Press Enter to exit.")
        return

    output_folder = os.path.join(folder_path, "split_images_output")
    os.makedirs(output_folder, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    
    files_to_process = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

    if not files_to_process:
        print(f"\nNo JPEG images found in the folder '{folder_path}' to split.")
        input("Press Enter to exit.")
        return

    for filename in files_to_process:
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size

                split_x = find_vertical_black_line_center(file_path, search_middle_fraction=0.2)
                
                if split_x is None:
                    split_x = width // 2
                    print(f"No clear line found in '{filename}'. Splitting at geometric center ({split_x}).")
                else:
                    print(f"Detected line in '{filename}' at x-coordinate: {split_x}. Splitting there.")

                split_x = max(1, min(split_x, width - 1)) 

                left_half = img.crop((0, 0, split_x, height))
                right_half = img.crop((split_x, 0, width, height))

                # Apply 180-degree rotation if requested by the user
                if flip_180:
                    left_half = left_half.rotate(180)
                    right_half = right_half.rotate(180)

                base_name, ext = os.path.splitext(filename)
                left_half_name = f"{base_name}(1){ext}"
                right_half_name = f"{base_name}(2){ext}"

                left_half.save(os.path.join(output_folder, left_half_name))
                right_half.save(os.path.join(output_folder, right_half_name))

                print(f"Saved '{left_half_name}' and '{right_half_name}'.")
                processed_count += 1

        except Exception as e:
            print(f"Could not process '{filename}'. Error: {e}")
            skipped_count += 1
    
    print("\n--- Processing Complete ---")
    print(f"Successfully split {processed_count} images.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images due to errors.")
    print(f"All split images are saved in the '{output_folder}' folder.")
    
    input("Press Enter to exit.")


if __name__ == "__main__":
    exe_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    print("-------------------------------------------------------------------")
    print("  JPEG Photo Splitter & Rotator")
    print("  This tool splits JPEG photos in half, looking for a dividing line.")
    print("-------------------------------------------------------------------")
    
    # 1. Ask user for the folder name
    while True:
        subfolder_name = input("\nEnter the NAME of the folder containing your photos (e.g., 'MyPhotos'):\n> ").strip()
        
        if not subfolder_name:
            print("Folder name cannot be empty. Please try again.")
            continue

        photos_folder_path = os.path.join(exe_directory, subfolder_name)
        
        if os.path.isdir(photos_folder_path):
            print(f"\nFound folder: '{photos_folder_path}'")
            break
        else:
            print(f"Error: Folder '{subfolder_name}' not found inside '{exe_directory}'.")
            print("Please make sure the photos folder is in the SAME location as this application.")
            
    # 2. Ask user if they want to flip the images 180 degrees
    while True:
        flip_input = input("\nDo you want to rotate every split photo 180 degrees? (yes/no or y/n):\n> ").strip().lower()
        if flip_input in ['yes', 'y']:
            rotate_images = True
            print("Rotation enabled: Images will be flipped 180 degrees.")
            break
        elif flip_input in ['no', 'n']:
            rotate_images = False
            print("Rotation disabled: Images will maintain original orientation.")
            break
        else:
            print("Invalid input. Please type 'yes' or 'no'.")

    print("\nStarting image splitting process...")
    split_jpeg_in_half_by_line(photos_folder_path, flip_180=rotate_images)