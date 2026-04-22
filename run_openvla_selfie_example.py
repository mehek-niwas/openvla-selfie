"""
run_openvla_selfie_example.py
=============================

End-to-end example: load the real OpenVLA-7B model from Hugging Face and ask it
to interpret its own hidden embeddings for a given (image, instruction) pair.

Requirements (these match OpenVLA's pins):
    pip install "torch==2.2.0" "transformers==4.40.1" "tokenizers==0.19.1" \
                "timm==0.9.10" Pillow accelerate
    # optional but recommended: pip install "flash-attn==2.5.5" --no-build-isolation

Hardware: ~16 GB of GPU VRAM in bf16 is enough for inference (no training here).

Note: This script requires internet + HF access to download the 15 GB OpenVLA
checkpoint. It is NOT run in this screening environment; the actual hook logic
is exercised in test_hooks_on_tiny_llama.py.
"""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from openvla_selfie import InterpretationPrompt, interpret_openvla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="Path to an RGB image.")
    ap.add_argument(
        "--instruction", type=str,
        default="pick up the red block and place it on the plate",
    )
    ap.add_argument("--model-id", type=str, default="openvla/openvla-7b")
    ap.add_argument("--inject-layer", type=int, default=2,
                    help="Layer k at which to overwrite placeholder hiddens.")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    ap.add_argument("--first-image-token-offset", type=int, default=1,
                    help="OpenVLA inserts image-patch embeddings after <BOS> (pos 1). "
                         "For prism-dinosiglip-224px there are 256 patch tokens.")
    ap.add_argument("--num-image-tokens", type=int, default=256)
    args = ap.parse_args()

    print(f"Loading {args.model_id} (this can take a while)...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    image = Image.open(args.image).convert("RGB")
    prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"

    # --- build a list of (layer, token) pairs to interpret ---
    # Pick 4 image-patch tokens spread across the visual grid (upper-left,
    # upper-right, lower-left, lower-right quadrants of the 16x16 = 256 grid)
    # plus the last text token (the one OpenVLA uses to produce the first
    # action token). The image patches live at positions
    #   [first_image_token_offset, first_image_token_offset + num_image_tokens).
    base = args.first_image_token_offset
    sqrt = int(args.num_image_tokens ** 0.5)
    patch_positions = [
        base + 0,                                    # top-left
        base + (sqrt - 1),                           # top-right
        base + sqrt * (sqrt - 1),                    # bottom-left
        base + sqrt * sqrt - 1,                      # bottom-right
        base + (sqrt // 2) * sqrt + (sqrt // 2),     # center
    ]
    text_last = base + args.num_image_tokens  # approximation; real seq_len is
                                              # returned by the processor
    # Build (layer, token) pairs: for each chosen layer, interpret all selected tokens.
    tokens_to_interpret = [(l, t) for l in args.layers
                           for t in patch_positions + [text_last]]

    # --- custom interpretation prompt (Llama-2 chat style works well on the
    # Llama-2 backbone of OpenVLA-7B) ---
    interp = InterpretationPrompt.build(
        processor.tokenizer,
        ("[INST] ", 0, 0, 0, 0, 0,
         " [/INST] This hidden state represents the concept:"),
    )

    print("Running SelfIE-style interpretation...")
    results = interpret_openvla(
        model, processor, image, prompt,
        tokens_to_interpret=tokens_to_interpret,
        interp_prompt=interp,
        inject_layer=args.inject_layer,
        max_new_tokens=args.max_new_tokens,
    )

    # --- pretty-print ---
    print(f"\nInstruction: {args.instruction}")
    print("=" * 78)
    for r in results:
        tok_label = r["token_decoded"]
        # label image patches specially for readability
        pos = r["token"]
        if base <= pos < base + args.num_image_tokens:
            grid_idx = pos - base
            row, col = grid_idx // sqrt, grid_idx % sqrt
            tok_label = f"<img[{row:2d},{col:2d}]>"
        print(f"  layer {r['layer']:>2d}  token {r['token']:>4d} "
              f"({tok_label:<18}) -> {r['interpretation']}")


if __name__ == "__main__":
    main()
