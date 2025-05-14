
# ShawGPT: Fine-Tuning Mistral 7B Instruct with YouTube Comments

ShawGPT is a fine-tuned version of the Mistral-7B-Instruct model, designed to act as a virtual data science consultant for YouTube. It responds to user comments naturally and contextually, adapting its tone and depth based on the input.

---

## 🧠 Model Objective

ShawGPT serves as a virtual assistant that:
- Uses accessible, beginner-friendly language for data science topics.
- Escalates to technical depth when required.
- Ends every response with its signature: `–ShawGPT`.
- Mirrors the tone and length of YouTube comments to create natural interactions.

---

## 📦 Base Model

- [`TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ)
- Quantized for efficiency with Auto-GPTQ

---

## 🔧 Setup

Install dependencies:

```
pip install optimum bitsandbytes auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft
```

---

## 🏋️‍♂️ Training Steps

1. **Load and prepare base model** using `AutoModelForCausalLM` and GPTQ.
2. **Tokenize prompts** using a structured `[INST]...[/INST]` format.
3. **Enable LoRA-based fine-tuning** using `peft`.
4. **Dataset:** `shawhin/shawgpt-youtube-comments` from Hugging Face.
5. **Train** using the `Trainer` API from Hugging Face's `transformers` with gradient checkpointing and 8-bit optimizations.

---

## 📁 Training Parameters

- Epochs: 10  
- Learning Rate: 2e-4  
- Batch Size: 4  
- Optimizer: `paged_adamw_8bit`  
- Gradient Accumulation: 4  
- FP16 Enabled  

---

## 🧪 Inference Examples

```python
comment = "What is self attention mechanism?"
prompt = f"""[INST] ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, thus keeping the interaction natural and engaging.
{comment} [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔄 Fine-Tuning Outputs

The final fine-tuned model is saved locally to `shawgpt-ft/` and can be pushed to the Hugging Face Hub for deployment.

---

## 📬 Feedback

For improvements, suggestions, or contributions, feel free to open an issue or pull request.

---

**License**: MIT  
