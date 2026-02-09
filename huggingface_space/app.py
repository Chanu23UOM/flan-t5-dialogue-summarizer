"""
Gradio app for Dialogue Summarization — runs on Hugging Face Spaces.

This file goes into a new Hugging Face Space (Gradio SDK).
It loads the fine-tuned FLAN-T5 LoRA adapter from the HF Hub and
lets users paste a conversation to get a summary.
"""

import torch
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ── CHANGE THIS to your uploaded model repo ───────────────────────────────────
MODEL_REPO = "YOUR_HF_USERNAME/flan-t5-dialogue-summarizer"
# ───────────────────────────────────────────────────────────────────────────────


def load_model():
    """Load base model + LoRA adapter from Hugging Face Hub."""
    config = PeftConfig.from_pretrained(MODEL_REPO)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


print("Loading model…")
model, tokenizer, device = load_model()
print(f"Model loaded on {device}")


def summarize(conversation: str) -> str:
    """Generate a summary for the given conversation."""
    if not conversation.strip():
        return "⚠️ Please enter a conversation."

    prompt = f"Summarize the following conversation:\n\n{conversation}\n\nSummary:"
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True
    ).to(device)

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


# ── Gradio Interface ──────────────────────────────────────────────────────────

SAMPLE = (
    "#Person1#: Hey, have you finished the project report yet?\n"
    "#Person2#: Almost. I just need to add the conclusion and proofread it.\n"
    "#Person1#: The deadline is tomorrow morning, right?\n"
    "#Person2#: Yes, I'll have it done by tonight. Can you review it after?\n"
    "#Person1#: Sure, just send it over when you're done.\n"
    "#Person2#: Great, I'll email it to you by 9 PM."
)

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        label="Conversation",
        placeholder="Paste your conversation here…",
        lines=12,
        value=SAMPLE,
    ),
    outputs=gr.Textbox(label="Summary", lines=4),
    title="📝 Dialogue Summarizer",
    description=(
        "Fine-tuned **FLAN-T5** with **LoRA** on the DialogSum dataset.  \n"
        "Paste any conversation and click **Submit** to get a summary."
    ),
    examples=[
        [SAMPLE],
        [
            "#Person1#: I'm thinking of upgrading my computer hardware.\n"
            "#Person2#: What kind of upgrades do you have in mind?\n"
            "#Person1#: I definitely need more RAM and maybe a better graphics card for gaming.\n"
            "#Person2#: Make sure your power supply can handle the new card."
        ],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
