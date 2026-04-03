import re
from collections import Counter
import pandas as pd

def analyze_dataset(file_path):
    tag_pattern = re.compile(r'^<([A-Z]+)>')
    tags = []
    line_lengths = []
    word_counts = []
    unique_tokens = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            match = tag_pattern.match(line)
            if match:
                tags.append(match.group(1))
            
            # Remove tag and structural markers for stats
            clean_text = re.sub(r'<[A-Z]+>\s*:\s*', '', line).replace('<eos>', '').strip()
            line_lengths.append(len(clean_text))
            
            words = clean_text.split()
            word_counts.append(len(words))
            unique_tokens.update(words)
            
    df_tags = pd.Series(tags).value_counts()
    
    print("--- Dataset Analysis ---")
    print(f"Total lines: {len(line_lengths)}")
    print(f"Tag distribution:\n{df_tags}")
    print(f"Average line length (chars): {sum(line_lengths)/len(line_lengths):.2f}")
    print(f"Average words per line: {sum(word_counts)/len(word_counts):.2f}")
    print(f"Total words: {sum(word_counts)}")
    print(f"Unique tokens: {len(unique_tokens)}")
    print(f"Top 20 tokens: {unique_tokens.most_common(20)}")

if __name__ == "__main__":
    file_path = r'c:\Users\admin\Downloads\VedSastra\Epic\data\processed\sanskrit_epic_dataset.txt'
    analyze_dataset(file_path)
