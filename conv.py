import os

# Path to your folder
folder_path = r"images"

# Get all image files
image_extensions = (".jpg", ".jpeg", ".png")
files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Sort files for consistent renaming
files.sort()

# Rename images sequentially
for index, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)

    # Keep original extension
    extension = os.path.splitext(filename)[1]

    new_filename = f"{index}{extension}"
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)

    print(f"Renamed: {filename} -> {new_filename}")

print("All images renamed successfully.")