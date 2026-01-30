
import os
import argparse
from transformers import AutoTokenizer

def build_chat(tokenizer, prompt, disable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    if disable_thinking:
        # 尝试传递 enable_thinking=False
        try:
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             return prompt
        except TypeError as e:
             # 如果 tokenizer 不支持 enable_thinking 参数，则回退或者报错
             print(f"Warning: Tokenizer does not support enable_thinking argument: {e}")
             # 回退到默认行为，但这样无法 disable thinking
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             return prompt
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return prompt

def test_disable_thinking(model_path):
    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    test_prompt = "Hello, who are you?"

    print("\n--- Testing WITHOUT disable_thinking (Default) ---")
    output_default = build_chat(tokenizer, test_prompt, disable_thinking=False)
    print("Output (repr):")
    print(repr(output_default))
    
    if "<think>" in output_default:
        print("Result: Default includes <think> tag (Expected for Qwen3-QwQ).")
    else:
        print("Result: Default does NOT include <think> tag.")

    print("\n--- Testing WITH disable_thinking=True ---")
    output_disabled = build_chat(tokenizer, test_prompt, disable_thinking=True)
    print("Output (repr):")
    print(repr(output_disabled))

    if "<think>\n\n</think>" in output_disabled:
         print("Result: SUCCESS. Found empty thinking block <think>\\n\\n</think>.")
    elif "<think>" not in output_disabled:
         print("Result: SUCCESS. No <think> tag found (fully removed).")
    else:
         print("Result: FAIL. <think> tag still present and not empty.")

if __name__ == "__main__":
    MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
    test_disable_thinking(MODEL_PATH)
