"""
try_on_tinyllama.py
===================

Run the SelfIE hook pipeline on TinyLlama-1.1B-Chat — a real trained model small
enough to fit on ~3 GB of VRAM. This is an intermediate validation step between
the random-init sanity test and the full OpenVLA run:

  * test_hooks_on_tiny_llama.py  -> proves the hooks are wired correctly
                                    (random weights, gibberish interpretations)
  * try_on_tinyllama.py          -> proves the pipeline produces coherent
                                    English on a real trained model
                                    (this file)
  * run_openvla_selfie_example.py-> the actual OpenVLA-7B demo (needs 16GB VRAM)

Usage:
    python try_on_tinyllama.py
    python try_on_tinyllama.py --prompt "The capital of France is"
    python try_on_tinyllama.py --prompt "..." --layers 4 10 16 20

First run downloads ~2.2 GB of weights to ~/.cache/huggingface.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from openvla_selfie import InterpretationPrompt, record_hidden_states, interpret_embedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 10, 16, 20])
    ap.add_argument("--inject-layer", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--token", type=int, default=None,
                    help="Token position to interpret (default: last token).")
    args = ap.parse_args()

    print(f"Loading {args.model_id} (first run downloads ~2.2 GB)...")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
    ).to("cuda:0").eval()
    print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  hidden_size       = {model.config.hidden_size}")
    print()

    # --- 1. Original forward pass: cache all layer hidden states ---
    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    all_hidden = record_hidden_states(model,
                                      input_ids=input_ids,
                                      attention_mask=attention_mask)

    target_tok = args.token if args.token is not None else input_ids.shape[1] - 1
    tok_str = tok.decode(input_ids[0, target_tok])
    print(f"Prompt: {args.prompt!r}")
    print(f"Interpreting token {target_tok} ({tok_str!r}) across layers {args.layers}")
    print(f"Injecting into layer {args.inject_layer} of the interpretation pass")
    print("=" * 78)

    # --- 2. Build the interpretation prompt ---
    interp = InterpretationPrompt.build(
        tok,
        ("[INST] ", 0, 0, 0, 0, 0, " [/INST] Sure, I will summarize the message:"),
    )

    # --- 3. Interpret the chosen token at each requested layer ---
    for layer in args.layers:
        vec = all_hidden[layer][0, target_tok]
        text = interpret_embedding(
            model, tok, vec, interp,
            inject_layer=args.inject_layer,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  layer {layer:>2d}: {text.strip()}")

    # --- 4. For contrast: what does the model actually predict next? ---
    print()
    print("For reference, the model's actual next-token prediction:")
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        top5 = logits[0, -1].topk(5)
        for prob_idx, token_id in zip(top5.values, top5.indices):
            print(f"  {tok.decode(token_id)!r:<20} logit={prob_idx.item():+.2f}")


if __name__ == "__main__":
    main()
