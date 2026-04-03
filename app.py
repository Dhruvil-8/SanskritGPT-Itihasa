import gradio as gr
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import os
import threading

# --- Configuration ---
MODEL_PATH = "model/Epic/sanskrit-gpt-epic-hyper"
DEFAULT_MAX_LENGTH = 512

# Global model state for lazy loading
model_state = {"model": None, "tokenizer": None}
model_lock = threading.Lock()

def load_model():
    """Thread-safe lazy loading of the model and tokenizer."""
    with model_lock:
        if model_state["model"] is None:
            print(f"Loading SanskritGPT-Itihasa from {MODEL_PATH}...")
            try:
                model_state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)
                model_state["model"] = GPT2LMHeadModel.from_pretrained(
                    MODEL_PATH, 
                    low_cpu_mem_usage=True
                ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Critial Error loading model: {e}")
                return None, None
    return model_state["model"], model_state["tokenizer"]

def clean_output(text):
    """
    Strips metadata tags and normalizes spaces for a professional-grade output.
    """
    # Remove metadata tags and common BPE artifacts
    tags = ["<MBH>", "<RAM>", "<eos>", "<pad>", "<bos>"]
    for tag in tags:
        text = text.replace(tag, "")
    
    # Normalize space markers used by GPT-2/BPE decoders
    # The '_' and 'Ġ' are common space-byte representations
    text = text.replace("Ġ", " ").replace("_", " ")
    
    # Clean up double spaces or leading/trailing whitespace
    text = " ".join(text.split())
    
    # Remove any stray colons or punctuation that might have bled through
    text = text.replace(" : ", " ").replace(" :", " ")
    
    return text.strip()

def generate_verse(prompt, style, temperature, top_p, max_length):
    model, tokenizer = load_model()
    
    if model is None:
        return "⚠️ Error: The model could not be loaded. Please ensure the 'sanskrit-gpt-epic-hyper' folder exists in the root directory."

    try:
        # Style-Specific Prompt Engineering
        if style == "Mahabharata":
            full_prompt = f"<MBH> {prompt}"
        elif style == "Ramayana":
            full_prompt = f"<RAM> {prompt}"
        else:  # Classical Synthesis
            full_prompt = prompt

        device = next(model.parameters()).device
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        
        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        return clean_output(raw_text)
    
    except Exception as e:
        return f"❌ Generation Error: {str(e)}"

# --- CUSTOM AESTHETICS ---
custom_css = """
body { background-color: #0d1117; color: #c9d1d9; font-family: 'Outfit', sans-serif; }
.gradio-container { border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
.main-header { text-align: center; color: #58a6ff; margin-bottom: 2rem; }
.footer-text { font-size: 0.85rem; color: #8b949e; margin-top: 2rem; }
"""

# --- UI BUILD ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("<div class='main-header'><h1>🕉️ SanskritGPT-Itihasa Explorer</h1><p>Hyper-Precision Transformer for Sanskrit Epics (Itihasa)</p></div>")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Verse Prompt (Devanagari)",
                placeholder="प्रारम्भः भवतु... (Enter starting words)",
                lines=3
            )
            style_selector = gr.Radio(
                choices=["Mahabharata", "Ramayana", "Classical Synthesis"],
                value="Classical Synthesis",
                label="Style Mode"
            )
            generate_btn = gr.Button("Generate Epic Verse", variant="primary")
            
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ Parameters", open=False):
                temp_slider = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-P")
                max_len_slider = gr.Slider(32, 512, value=256, step=32, label="Max Length")
            
    output_display = gr.Textbox(
        label="Generated Sanskrit Verse",
        lines=6,
        interactive=False,
        placeholder="Output will appear here..."
    )
    
    generate_btn.click(
        fn=generate_verse,
        inputs=[input_text, style_selector, temp_slider, top_p_slider, max_len_slider],
        outputs=output_display
    )

    with gr.Row(elem_classes="footer-text"):
        with gr.Column():
            gr.Markdown("### 🧠 Model Info")
            gr.Markdown("- **Architecture:** GPT-2 (512/8/8)\n- **Context Window:** 512 Tokens\n- **Tuning:** Hyper-Precision Epic (30 Epochs)")
        with gr.Column():
            gr.Markdown("### ⚖️ Research Disclaimer")
            gr.Markdown("> ⚠️ This project is a computational linguistic experiment.\n> - Generated text is **not authentic scripture**.\n> - Purpose: Structural and Narrative Research.\n> - For academic and research use only.")

if __name__ == "__main__":
    demo.launch()
