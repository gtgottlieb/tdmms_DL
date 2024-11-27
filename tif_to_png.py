import os
from PIL import Image
ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../'))

# Edit this to necessary .tif file directory
tif_folder = os.path.abspath(os.path.join('data/images/batch1'))
print(tif_folder)

os.makedirs(tif_folder, exist_ok=True)

tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith((".tif", ".tiff"))]
total_files = len(tif_files)
print(f"Total .tif files found: {total_files}")

if total_files > 0:
    for index, filename in enumerate(tif_files, start=1):
        input_path = os.path.join(tif_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(tif_folder, output_filename)

        try:
            with Image.open(input_path) as img:
                img.save(output_path, "PNG")
                print(f"Converted {index}/{total_files} files: {filename} -> {output_filename}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
            tif_files.remove(filename)
    print("Conversion complete.")

    # Ask if the user wants to delete the original .tif files
    delete = input("Do you want to delete the original .tif files? (yes/no): ").strip().lower()
    if delete in ['yes', 'y']:
        for filename in tif_files:
            try:
                os.remove(os.path.join(tif_folder, filename))
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")
    else: 
        print(f"No items deleted.")

else:
    print("No .tif files found in the input folder.")

