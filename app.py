"""
Streamlit app for Dialogue Summarization using fine-tuned FLAN-T5 + LoRA.

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR = "./flan-t5-lora-dialogue-summary"   # path to saved LoRA adapter


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the base model + LoRA adapter once and cache it."""
    config = PeftConfig.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_summary(model, tokenizer, dialogue: str, device: str) -> str:
    """Generate a summary for a given dialogue."""
    prompt = f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dialogue Summarizer", page_icon="📝", layout="centered")
st.title("📝 Dialogue Summarizer")
st.caption("Powered by FLAN-T5 + LoRA fine-tuned on DialogSum")

model, tokenizer, device = load_model()

# Sample conversation for quick testing
SAMPLE = (
    "#Person1#: Hey, have you finished the project report yet?\n"
    "#Person2#: Almost. I just need to add the conclusion and proofread it.\n"
    "#Person1#: The deadline is tomorrow morning, right?\n"
    "#Person2#: Yes, I'll have it done by tonight. Can you review it after?\n"
    "#Person1#: Sure, just send it over when you're done.\n"
    "#Person2#: Great, I'll email it to you by 9 PM."
)

conversation = st.text_area(
    "Paste a conversation below:",
    value=SAMPLE,
    height=250,
    help="Use #Person1#: and #Person2#: format, or any free-form conversation.",
)

if st.button("Summarize", type="primary"):
    if not conversation.strip():
        st.warning("Please enter a conversation first.")
    else:
        with st.spinner("Generating summary…"):
            summary = generate_summary(model, tokenizer, conversation, device)
        st.subheader("Summary")
        st.success(summary)
