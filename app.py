import gradio as gr
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import os
import threading

# --- Configuration ---
MODEL_PATH = "model/Epic/sanskrit-gpt-epic-hyper"
DEFAULT_MAX_LENGTH = 512

# Global model state for lazy loading
model_state = {"model": None, "tokenizer": None, "loaded": False, "error": None}
model_lock = threading.Lock()

def load_model():
    """Thread-safe lazy loading of the model and tokenizer."""
    with model_lock:
        if model_state["model"] is None and model_state["error"] is None:
            print(f"Loading SanskritGPT-Itihasa from {MODEL_PATH}...")
            try:
                model_state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)
                model_state["model"] = GPT2LMHeadModel.from_pretrained(
                    MODEL_PATH,
                    low_cpu_mem_usage=True
                ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
                model_state["loaded"] = True
                print("Model loaded successfully.")
            except Exception as e:
                model_state["error"] = str(e)
                print(f"Critical Error loading model: {e}")
                return None, None
    return model_state["model"], model_state["tokenizer"]

def clean_output(text):
    """
    Cleans BPE/Metaspace artifacts and joins Sanskrit suffixes to their base words.
    """
    # Restore Metaspace marker to standard space
    text = text.replace("\u2581", " ")

    # Sanskrit Suffix Joining: Remove spaces before combining marks and case endings
    suffixes = [
        " ं", " ः", " ा", " ि", " ी", " ु", " ू", " े", " ै", " ो", " ौ", " ्",
        " ेषु", " ाः", " ान", " स्य", " स्तु", " न्"
    ]
    for s in suffixes:
        text = text.replace(s, s.strip())

    # Final cleanup
    text = " ".join(text.split())
    return text.strip()

def generate_verse(prompt, style, temperature, top_p, max_length):
    if not prompt or not prompt.strip():
        return "कृपया प्रारम्भिक शब्दाः लिखन्तु — Please enter a starting prompt."

    model, tokenizer = load_model()

    if model is None:
        error_msg = model_state.get("error", "Unknown error")
        return f"⚠️ Model Loading Error: {error_msg}"

    try:
        # Style-Specific Prompt Engineering
        if style == "महाभारत (Mahabharata)":
            full_prompt = f"<MBH> {prompt}"
        elif style == "रामायण (Ramayana)":
            full_prompt = f"<RAM> {prompt}"
        else:
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

        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return clean_output(raw_text)

    except Exception as e:
        return f"❌ Generation Error: {str(e)}"


# --- PREMIUM THEME CSS ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');

/* === Global === */
body, .gradio-container {
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif !important;
    background: linear-gradient(145deg, #0a0e1a 0%, #111827 50%, #0f172a 100%) !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* === Hero Header === */
.hero-section {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(217, 119, 6, 0.08) 0%, rgba(15, 23, 42, 0.4) 100%);
    border-bottom: 1px solid rgba(217, 119, 6, 0.15);
    border-radius: 16px 16px 0 0;
    margin-bottom: 0.5rem;
}
.hero-section h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 40%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}
.hero-section .subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.hero-section .badge {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 4px 14px;
    background: rgba(217, 119, 6, 0.12);
    border: 1px solid rgba(217, 119, 6, 0.25);
    border-radius: 20px;
    font-size: 0.75rem;
    color: #fbbf24;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* === Glass Card === */
.glass-card {
    background: rgba(30, 41, 59, 0.5) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
}

/* === Input/Output Fields === */
textarea, input[type="text"] {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Noto Sans Devanagari', 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: rgba(245, 158, 11, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.1) !important;
    outline: none !important;
}

/* === Generate Button === */
.generate-btn {
    background: linear-gradient(135deg, #d97706 0%, #f59e0b 50%, #d97706 100%) !important;
    background-size: 200% 200% !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.4s ease !important;
    box-shadow: 0 4px 15px rgba(217, 119, 6, 0.25) !important;
}
.generate-btn:hover {
    background-position: right center !important;
    box-shadow: 0 6px 25px rgba(217, 119, 6, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* === Output Box === */
.output-box textarea {
    background: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(245, 158, 11, 0.15) !important;
    font-family: 'Noto Sans Devanagari', serif !important;
    font-size: 1.15rem !important;
    line-height: 1.9 !important;
    color: #fef3c7 !important;
    padding: 1rem !important;
}

/* === Sidebar Controls === */
.sidebar-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

/* === Radio & Slider Styles === */
.gradio-radio label {
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
    background: rgba(15, 23, 42, 0.4) !important;
}
.gradio-radio label:hover {
    border-color: rgba(245, 158, 11, 0.3) !important;
    background: rgba(217, 119, 6, 0.06) !important;
}
.gradio-radio input:checked + label {
    border-color: #f59e0b !important;
    background: rgba(217, 119, 6, 0.1) !important;
}

/* === Footer === */
.footer-section {
    margin-top: 1.5rem;
    padding: 1.25rem;
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 12px;
}
.footer-section h3 {
    font-size: 0.85rem;
    color: #f59e0b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.footer-section p, .footer-section li {
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.6;
}

/* === Quick Prompt Examples === */
.example-btn {
    font-size: 0.85rem !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    color: #cbd5e1 !important;
    transition: all 0.2s ease !important;
}
.example-btn:hover {
    border-color: rgba(245, 158, 11, 0.3) !important;
    color: #fbbf24 !important;
}

/* === Accordion === */
.gradio-accordion {
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 10px !important;
    background: transparent !important;
}
"""


# --- UI BUILD ---
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=custom_css,
    title="SanskritGPT-Itihasa | Sanskrit Epic Verse Generator"
) as demo:

    # --- Hero Header ---
    gr.HTML("""
    <div class="hero-section">
        <h1>🕉 SanskritGPT-Itihasa</h1>
        <p class="subtitle">Hyper-Precision Transformer for Sanskrit Epic Verse Generation</p>
        <span class="badge">GPT-2 · 42M PARAMS · TRAINED ON DEVANAGARI EPICS</span>
    </div>
    """)

    # --- Main Layout ---
    with gr.Row():

        # --- Left: Sidebar Controls ---
        with gr.Column(scale=1, min_width=260):
            gr.HTML('<div class="sidebar-title">⚙ Generation Settings</div>')

            style_selector = gr.Radio(
                choices=["महाभारत (Mahabharata)", "रामायण (Ramayana)", "शास्त्रीय (Classical)"],
                value="शास्त्रीय (Classical)",
                label="Epic Style",
                info="Select the literary tradition to guide generation."
            )

            temp_slider = gr.Slider(
                0.1, 1.5, value=0.8, step=0.05,
                label="Temperature",
                info="Lower = focused, Higher = creative"
            )
            top_p_slider = gr.Slider(
                0.5, 1.0, value=0.95, step=0.01,
                label="Top-P (Nucleus Sampling)",
                info="Controls diversity of word choices"
            )
            max_len_slider = gr.Slider(
                32, 512, value=256, step=32,
                label="Max Output Length",
                info="Maximum tokens to generate"
            )

            # --- Model Specifications ---
            with gr.Accordion("📊 Model Specifications", open=False):
                gr.Markdown("""
| Specification | Value |
|:---|:---|
| **Architecture** | GPT-2 (Custom) |
| **Parameters** | ~42 Million |
| **Layers / Heads** | 8 / 8 |
| **Context Window** | 512 tokens |
| **Tokenizer** | Unigram (Metaspace) |
| **Training** | 30 epochs, T4 GPU |
| **Corpus** | Mahabharata + Ramayana |
                """)

        # --- Right: Generation Workspace ---
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Sanskrit Verse Prompt (Devanagari)",
                placeholder="प्रारम्भः भवतु... (Enter starting words in Devanagari)",
                lines=3,
                max_lines=5,
            )

            # Quick Prompts
            gr.Examples(
                examples=[
                    ["धर्मक्षेत्रे कुरुक्षेत्रे"],
                    ["तपस्वी च महातेजाः"],
                    ["राजा दशरथो"],
                    ["अर्जुन उवाच"],
                    ["भीष्म उवाच धर्मराज"],
                ],
                inputs=input_text,
                label="⚡ Quick Prompts",
            )

            generate_btn = gr.Button(
                "✦  Generate Epic Verse",
                variant="primary",
                elem_classes=["generate-btn"],
                size="lg"
            )

            output_display = gr.Textbox(
                label="Generated Sanskrit Verse",
                lines=8,
                max_lines=15,
                interactive=False,
                placeholder="Generated verse will appear here...",
                elem_classes=["output-box"],
            )

    # --- Footer ---
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="footer-section">
                <h3>⚖ Research Disclaimer</h3>
                <p>This is a computational linguistic experiment. Generated text is <strong>not authentic scripture</strong>.
                The model was trained from scratch on Devanagari text of the Mahabharata and Ramayana.
                For academic and research use only.</p>
            </div>
            """)
        with gr.Column():
            gr.HTML("""
            <div class="footer-section">
                <h3>🔬 About This Project</h3>
                <p>SanskritGPT-Itihasa is an AI-assisted model training and coding experiment.
                All training data, logs, notebooks, and model weights are open-sourced for reproducibility.
                Trained on Google Colab T4 GPU from scratch.</p>
            </div>
            """)

    # --- Event Binding ---
    generate_btn.click(
        fn=generate_verse,
        inputs=[input_text, style_selector, temp_slider, top_p_slider, max_len_slider],
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch()
