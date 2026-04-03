import os
import requests
import time

# --- CONFIGURATION ---
# The exact paths you provided
MBH_BASE_URL = "https://bombay.indology.info/mahabharata/text/UD/"
RAM_BASE_URL = "https://bombay.indology.info/ramayana/text/UD/"

# Folder names to be created
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MBH_FOLDER = os.path.join(base_dir, "data", "raw", "mahabharata")
RAM_FOLDER = os.path.join(base_dir, "data", "raw", "ramayana")

def download_raw(url_base, prefix, total_books, folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

    for i in range(1, total_books + 1):
        # Format book number: 1 becomes '01', 2 becomes '02'
        book_num = str(i).zfill(2)
        filename = f"{prefix}{book_num}.txt"
        url = f"{url_base}{filename}"
        
        save_path = os.path.join(folder_name, filename)

        print(f"Downloading: {url} ...")
        try:
            response = requests.get(url)
            # Check if the file exists on the server
            if response.status_code == 200:
                # Save the content EXACTLY as it is (binary write)
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"Saved to: {save_path}")
            else:
                print(f"Skipped: Book {book_num} not found (Status 404)")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
        
        # Pause for 1 second so we don't overwhelm the website
        time.sleep(1)

# --- START DOWNLOADS ---

# 1. Download Mahabharata (18 Parvas: MBh01.txt to MBh18.txt)
download_raw(MBH_BASE_URL, "MBh", 18, MBH_FOLDER)

# 2. Download Ramayana (7 Kandas: Ram01.txt to Ram07.txt)
download_raw(RAM_BASE_URL, "Ram", 7, RAM_FOLDER)

print("\n--- FINISHED ---")
print(f"Check the folders '{MBH_FOLDER}' and '{RAM_FOLDER}' for your raw files.")