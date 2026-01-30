
from transformers import AutoTokenizer

model_path = "/root/autodl-tmp/models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [{"role": "user", "content": "Hello"}]

print("--- Default ---")
prompt_default = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(prompt_default))

print("\n--- With enable_thinking=False ---")
try:
    prompt_nothink = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(repr(prompt_nothink))
except Exception as e:
    print(f"Error: {e}")
