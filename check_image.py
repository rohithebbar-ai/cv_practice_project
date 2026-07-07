import os
from PIL import Image

# UPDATE THIS to the exact path of your cropped test file
FILE_PATH = "local_uploads/your_cropped_file.jpg"

print("=== FILE CHECK ===")
print("File exists:", os.path.exists(FILE_PATH))

if os.path.exists(FILE_PATH):
    print("File size:", os.path.getsize(FILE_PATH), "bytes")

    img = Image.open(FILE_PATH)
    print("Actual format:", img.format)
    print("Dimensions (width x height):", img.size)
    print("Color mode:", img.mode)
else:
    print("!!! File not found at this path. Double check FILE_PATH above. !!!")
