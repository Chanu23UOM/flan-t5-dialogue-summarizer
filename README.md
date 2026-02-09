# Dialogue Summarizer — Fine-Tuned FLAN-T5 with LoRA

A dialogue summarization system built by fine-tuning Google's FLAN-T5 model using LoRA (Low-Rank Adaptation) on the DialogSum dataset. The model generates concise summaries from multi-turn conversations.

## Live Demo

Try the app on [Hugging Face Spaces](https://huggingface.co/spaces) or via the [GitHub Pages site](https://pages.github.com).

## Project Structure

```
.
├── flan_t5_lora_finetuning.ipynb      # Training notebook (LoRA fine-tuning pipeline)
├── Lab_2_fine_tune_generative_ai_model.ipynb  # Reference lab (full fine-tuning vs PEFT)
├── app.py                             # Streamlit app for local inference
├── upload_to_hf.py                    # Script to upload model to Hugging Face Hub
├── huggingface_space/
│   ├── app.py                         # Gradio app deployed on HF Spaces
│   ├── requirements.txt
│   └── README.md                      # HF Space metadata
├── docs/
│   └── index.html                     # GitHub Pages site (embeds HF Space)
└── flan-t5-lora-dialogue-summary/     # Saved LoRA adapter weights
```

## Approach

- **Base Model**: google/flan-t5-base (248M parameters)
- **Fine-Tuning Method**: LoRA via Hugging Face PEFT — only 0.6% of parameters are trainable
- **Dataset**: knkarthick/dialogsum (1,000 training samples, 1 epoch)
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum

### LoRA Configuration

| Parameter      | Value        |
|----------------|--------------|
| Rank (r)       | 16           |
| Alpha          | 32           |
| Dropout        | 0.05         |
| Target Modules | q, v         |
| Learning Rate  | 1e-3         |
| Batch Size     | 8            |

## Tech Stack

- Python, PyTorch
- Hugging Face Transformers, PEFT, Datasets, Evaluate
- Streamlit (local app), Gradio (HF Spaces)
- GitHub Pages (static site)

## License

This project is for educational purposes. The base model (FLAN-T5) is licensed under Apache 2.0.

