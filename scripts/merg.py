import os
import re

def clean_text(text):
    """
    Removes Latin characters, digits, and extra whitespace.
    Keeps only Devanagari and Sanskrit punctuation.
    """
    # Remove Latin letters (a-z, A-Z) and digits (0-9)
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_epics(config, output_filename):
    print("Starting Clean and Merge process...")
    total_verses = 0

    with open(output_filename, "w", encoding="utf-8") as outfile:
        for folder, tag in config.items():
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} not found. Skipping.")
                continue

            # Get list of files in order (MBh01, MBh02...)
            files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
            print(f"Processing {len(files)} files in {folder}...")

            for filename in files:
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as infile:
                    lines = infile.readlines()
                    
                    cleaned_lines = []
                    for line in lines:
                        # The website format is: [ID] [Text]
                        # Example: 01001001a nārāyaṇaṃ namaskṛtya
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) >= 2:
                            text_content = clean_text(parts[1])
                            if text_content:
                                cleaned_lines.append(text_content)
                    
                    # Pair the lines into full verses (Shlokas)
                    # We take line 1 and line 2, join them, then line 3 and 4...
                    for i in range(0, len(cleaned_lines) - 1, 2):
                        line1 = cleaned_lines[i]
                        line2 = cleaned_lines[i+1]
                        
                        # Format: <TAG> Part1 । Part2 ॥ <eos>
                        formatted_verse = f"{tag} {line1} । {line2} ॥ <eos>\n"
                        outfile.write(formatted_verse)
                        total_verses += 1

    print(f"\nSUCCESS!")
    print(f"Total verses processed: {total_verses}")
    print(f"Final training file saved as: {output_filename}")

# --- EXECUTION ---

# Define the folders and their corresponding tags
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
epic_config = {
    os.path.join(base_dir, "data", "raw", "mahabharata"): "<MBH>",
    os.path.join(base_dir, "data", "raw", "ramayana"): "<RAM>"
}

# The name of your final dataset file
FINAL_FILE = os.path.join(base_dir, "data", "processed", "sanskrit_epic_dataset.txt")

process_epics(epic_config, FINAL_FILE)