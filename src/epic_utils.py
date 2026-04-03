import re
import os
import pandas as pd
import numpy as np
from collections import Counter

def configure_sanskrit_plots():
    """
    Configures Matplotlib to render Devanagari characters using Noto Sans Devanagari.
    Falls back to sans-serif if the font is not found.
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    font_path = r'C:\Users\admin\AppData\Local\Microsoft\Windows\Fonts\NotoSansDevanagari.ttf'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        print(f"Devanagari font loaded: {prop.get_name()}")
    else:
        print('Devanagari font not found at specified path, falling back to sans-serif.')
        plt.rcParams['font.family'] = 'sans-serif'

def count_syllables(word):
    """
    Estimates the number of syllables in a Devanagari word.
    Heuristic: Counts explicit vowels and inherent 'a' in consonants (unless virama is present).
    """
    vowels = r'[अआइईउऊऋॠऌएऐओऔािीुूृॄेैोौ]'
    consonants = r'[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह]'
    virama = r'\u094D'
    
    explicit_vowels = len(re.findall(vowels, word))
    cons_count = len(re.findall(consonants, word))
    virama_count = len(re.findall(virama, word))
    
    # Inherent a = consonants - viramas
    # Estimated syllables = explicit vowels + remaining consonants
    estimated_syllables = explicit_vowels + (cons_count - virama_count)
    return max(1, estimated_syllables)

def calculate_distinction_scores(target_counts, reference_counts, epsilon=0.01):
    """
    Calculates distinction scores for words in a target set compared to a reference set.
    Score = freq_target / max(freq_reference, epsilon)
    """
    total_target = sum(target_counts.values())
    total_ref = sum(reference_counts.values())
    
    scores = {}
    for word, count in target_counts.items():
        freq_target = count / total_target
        freq_ref = reference_counts.get(word, 0) / total_ref
        scores[word] = freq_target / max(freq_ref, epsilon)
        
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

def build_character_network(df, characters, window_size=5):
    """
    Builds a weighted co-occurrence network for characters.
    Iterates through text and connects characters appearing within the window_size.
    """
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(characters)
    
    for text in df['clean_text']:
        tokens = text.split()
        present = [c for c in characters if c in tokens]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                if G.has_edge(present[i], present[j]):
                    G[present[i]][present[j]]['weight'] += 1
                else:
                    G.add_edge(present[i], present[j], weight=1)
    return G

# --- CONFIGURATION (Common Characters & Places) ---

GAZETTEER = {
    "अयोध्या": [26.7913, 82.1998],  # Ayodhya
    "मिथिला": [26.7111, 85.9250],   # Mithila (Janakpur)
    "हस्तिनापुर": [29.1711, 78.0201], # Hastinapura
    "इन्द्रप्रस्थ": [28.6139, 77.2090], # Indraprastha (Delhi)
    "कुरुक्षेत्र": [29.9691, 76.8198], # Kurukshetra
    "द्वारका": [22.2442, 68.9685],   # Dwaraka
    "लङ्का": [7.8731, 80.7718],      # Lanka (modern Sri Lanka proxy)
    "पञ्चाल": [28.2612, 79.1556],   # Panchala (Kampilya region)
    "मद्र": [32.1275, 74.3312],     # Madra (Sialkot region)
    "मत्स्य": [27.5624, 76.6226],   # Matsya (Alwar region)
    "मथुरा": [27.4924, 77.6737],    # Mathura
    "किष्किन्धा": [15.3353, 76.4600], # Kishkindha (Hampi region)
    "काशी": [25.3176, 82.9739],     # Kashi (Varanasi)
}

# Unified Character Lists
CORE_CHARACTERS = {
    'RAM': ["राम", "सीता", "लक्ष्मण", "हनुमान", "रावण", "सुग्रीव", "बाली", "भरत", "लङ्का"],
    'MBH': ["कृष्ण", "अर्जुन", "युधिष्ठिर", "भीम", "दुर्योधन", "कर्ण", "भीष्म", "द्रोण", "द्रौपदी", "शकुनि"]
}

# --- CORE UTILITIES ---

def clean_sanskrit_text(text):
    """
    Cleans text to keep only Devanagari characters and standard punctuation.
    """
    if not text: return ""
    # Keep Devanagari range, dandas, and spaces
    text = re.sub(r'[a-zA-Z0-9\(\)\[\]]', '', text)
    text = re.sub(r'[^ \u0900-\u097F।॥]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_id(id_str):
    """
    Robustly parses IDs like 01001001a or 1001001a into components.
    Uses reverse-indexing to handle variable-length book numbers.
    """
    # 1. Separate digits from non-digit suffix (pada)
    match = re.match(r'(\d+)([a-zA-Z]*)', id_str)
    if not match: return None
    
    digits = match.group(1)
    pada = match.group(2)
    
    # 2. Extract components based on 3-digit sloka and 3-digit chapter
    # Expectation: [BOOK] [CHAPTER:3] [SLOKA:3]
    if len(digits) < 7: return None # Minimum format: B CCC SSS
    
    try:
        sloka = int(digits[-3:])
        chapter = int(digits[-6:-3])
        book = int(digits[:-6])
        return book, chapter, sloka, pada
    except:
        return None


def load_epic_dataset(folder_path):
    """
    Loads all .txt files in a folder and returns a structured DataFrame.
    """
    data = []
    if not os.path.exists(folder_path):
        print(f"Warning: Path {folder_path} does not exist.")
        return pd.DataFrame()
        
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("%") or not line: continue
                
                parts = line.split(maxsplit=1)
                if len(parts) < 2: continue
                
                id_str = parts[0]
                text = parts[1].strip()
                
                meta = parse_id(id_str)
                if meta:
                    data.append({
                        "id": id_str,
                        "book": meta[0],
                        "chapter": meta[1],
                        "sloka": meta[2],
                        "pada": meta[3],
                        "text": text,
                        "clean_text": clean_sanskrit_text(text),
                        # Create a global unique sloka identifier
                        "sloka_id": f"{meta[0]:02d}_{meta[1]:03d}_{meta[2]:03d}"
                    })
    return pd.DataFrame(data)

# --- ANALYTICS HELPERS ---

def get_speaker_distribution(df):
    """
    Extracts speakers from lines containing 'उवाच'.
    """
    pattern = r'([\u0900-\u097F]+)\s+(उवाच)'
    speakers = df['clean_text'].str.extract(pattern)[0].dropna()
    return speakers.value_counts()

def calculate_shannon_entropy(labels):
    """
    Calculates Shannon entropy for a list of tokens.
    """
    from collections import Counter
    counts = Counter(labels)
    probs = np.array(list(counts.values())) / len(labels)
    return -np.sum(probs * np.log2(probs))

def get_geographic_data(locations):
    """
    Returns a DataFrame with coordinates for plotting.
    """
    found = []
    for loc in locations:
        if loc in GAZETTEER:
            found.append({"Location": loc, "lat": GAZETTEER[loc][0], "lon": GAZETTEER[loc][1]})
    return pd.DataFrame(found)
