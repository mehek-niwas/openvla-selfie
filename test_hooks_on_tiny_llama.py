"""
test_hooks_on_tiny_llama.py
===========================

Sanity test for the hook-based SelfIE-style injector.
Runs WITHOUT GPU and WITHOUT any huge checkpoint download: we build a tiny
random-initialized Llama model and verify:

  1. record_hidden_states returns one tensor per decoder layer with the right
     shape.
  2. A modifying forward_pre_hook injected at layer k actually changes the
     downstream hidden state at layer k+1 (i.e. our injection really reaches
     the computation graph).
  3. interpret_embedding completes a generation pass end-to-end with an
     embedding injected at the placeholder positions of an interpretation
     prompt (we don't care that the tokens are gibberish -- the model is random
     -- we only care that the pipeline runs, the shape arithmetic is right,
     and the interpretation changes when we change the injected vector).

Run:
    python test_hooks_on_tiny_llama.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

from openvla_selfie import (
    InterpretationPrompt,
    record_hidden_states,
    interpret_embedding,
    _get_llama_layers,
    _Injector,
)


def build_tiny_llama():
    """Tiny Llama for fast local testing. Uses Llama-2's tokenizer so the
    [INST] / [/INST] tags tokenize the same way they would for OpenVLA's
    Llama-2 backbone.
    """
    cfg = LlamaConfig(
        vocab_size=32000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    # Use a real Llama-2 tokenizer so the interpretation prompt tokenizes
    # realistically. Fallback to GPT-2 if we can't download Llama-2.
    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
    return model, tok


def test_record_shape(model, tok):
    print("[1] test_record_shape ...", end=" ", flush=True)
    ids = tok("hello world, this is a test sentence", return_tensors="pt").input_ids
    hs_list = record_hidden_states(model, input_ids=ids, attention_mask=torch.ones_like(ids))
    assert len(hs_list) == model.config.num_hidden_layers, \
        f"got {len(hs_list)} hidden states, expected {model.config.num_hidden_layers}"
    for i, h in enumerate(hs_list):
        assert h.shape == (1, ids.shape[1], model.config.hidden_size), \
            f"layer {i} shape {h.shape} unexpected"
    print("OK")
    return hs_list


def test_injection_actually_changes_downstream(model, tok):
    print("[2] test_injection_actually_changes_downstream ...", end=" ", flush=True)
    ids = tok("hello world, this is a test sentence", return_tensors="pt").input_ids
    layers = _get_llama_layers(model)

    # original layer-2 hidden state
    hs_orig = record_hidden_states(model, input_ids=ids, attention_mask=torch.ones_like(ids))
    layer_k = 2
    pos = [3, 4]

    # now inject a zero vector at layer_k at positions pos and re-record layer k+1
    D = model.config.hidden_size
    zero_vec = torch.zeros(D, dtype=hs_orig[0].dtype)

    injector = _Injector(layers)
    injector.inject_at = {layer_k: (pos, zero_vec)}

    captured = {}
    def capture_next(_m, inputs):
        captured["hs"] = inputs[0].detach().clone()
    handle = layers[layer_k + 1].register_forward_pre_hook(capture_next)
    inject_handle = layers[layer_k].register_forward_pre_hook(
        injector._make_inject_hook(layer_k)
    )
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    handle.remove()
    inject_handle.remove()

    # the captured hidden state (entering layer_k+1) should differ from the
    # original run because layer_k's input was overwritten at two positions.
    # Note the layer-k ouput != its input; changing the input changes the output
    # everywhere the attention mixes those positions in.
    assert not torch.allclose(captured["hs"], hs_orig[layer_k + 1], atol=1e-5), \
        "injection did NOT propagate to layer_k+1; hook is not wired correctly"
    print("OK")


def test_end_to_end_interpretation(model, tok):
    print("[3] test_end_to_end_interpretation ...", end=" ", flush=True)
    ip = InterpretationPrompt.build(
        tok,
        ("[INST] ", 0, 0, 0, 0, 0, " [/INST] Sure, I will summarize the message:"),
    )
    assert len(ip.placeholder_positions) > 0, "no placeholders were located"
    # build a fake embedding and run the full interpret_embedding pipeline
    torch.manual_seed(1)
    vec_a = torch.randn(model.config.hidden_size)
    vec_b = torch.randn(model.config.hidden_size) * 10.0

    out_a = interpret_embedding(model, tok, vec_a, ip, inject_layer=1, max_new_tokens=5)
    out_b = interpret_embedding(model, tok, vec_b, ip, inject_layer=1, max_new_tokens=5)
    # With a random-init model the outputs are gibberish, but they should
    # (almost surely) differ when we inject different vectors -- this is what
    # would let SelfIE discriminate between embeddings.
    print(f"\n     injected vec_a -> '{out_a}'")
    print(f"     injected vec_b -> '{out_b}'")
    assert out_a != out_b, "same output for different injected vectors -- hook is a no-op!"
    print("[3] OK")


def main():
    print("Building tiny Llama (random init, 4 layers, hidden_size=64)...")
    model, tok = build_tiny_llama()
    print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
    print(f"  hidden_size       = {model.config.hidden_size}")
    print()

    test_record_shape(model, tok)
    test_injection_actually_changes_downstream(model, tok)
    test_end_to_end_interpretation(model, tok)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
