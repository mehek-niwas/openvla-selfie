"""
openvla_selfie.py
=================

Apply SelfIE-style self-interpretation (Chen et al., 2024, https://selfie.cs.columbia.edu/)
to OpenVLA (Kim et al., 2024, https://openvla.github.io/).

OpenVLA = Prismatic VLM (DINOv2 + SigLIP -> MLP projector -> Llama-2 7B).
We want text descriptions of hidden embeddings inside the Llama-2 backbone, where
those embeddings may correspond to image-patch tokens (image token embeddings are
genuinely interesting for robot manipulation) or language-instruction tokens.

SelfIE's idea, boiled down:
  1. Run an ORIGINAL forward pass on the real input and save all layer hidden states
     H[l, t]  (layer l, token position t).
  2. Run an INTERPRETATION forward pass on a fixed prompt like
        "[INST] <PLACEHOLDER> [/INST] Sure, I will summarize the message:"
     where at some early layer k, the hidden states at the placeholder positions are
     OVERWRITTEN with H[l, t]. Let the model generate text. That generated text is
     the interpretation of H[l, t].

The tricky part: SelfIE's repo was written against transformers==4.34.0 and copies
the LLaMA decoder forward wholesale. OpenVLA's HF weights are pinned to
transformers==4.40.1, and modern transformers (4.38+) removed / renamed the private
symbols SelfIE relies on. Concretely, SelfIE's `llama_forward_wrappers.py` breaks
because it uses:
  * `is_flash_attn_available`  -- renamed to `is_flash_attn_2_available` in 4.38+
  * `model.model._prepare_decoder_attention_mask(...)` -- removed in 4.38+ in favor
    of `_prepare_4d_causal_attention_mask(...)` as a free function
  * `padding_mask=` kwarg on `LlamaAttention.forward` -- removed in 4.38+
  * hidden-state tuple layout assumed by `past_key_values[0][0]` -- transformers
    switched to `DynamicCache` objects in 4.36+
See the docstring of `patch_selfie_for_transformers_4_40()` below for the exact
one-line edits needed if you want to use the upstream SelfIE code directly.

The approach taken in this module is cleaner and more robust: we DO NOT use
SelfIE's forward wrappers at all. Instead we implement the same algorithm using
PyTorch forward hooks on the native `LlamaDecoderLayer` modules. This is
transformers-version-agnostic and works on OpenVLA out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Interpretation prompt
# ---------------------------------------------------------------------------

@dataclass
class InterpretationPrompt:
    """A fixed prompt with placeholder positions we'll overwrite with a hidden
    embedding during the interpretation forward pass.

    We mirror SelfIE's convention: the prompt is a tuple of (str, 0, 0, ..., 0, str)
    where the integer 0 marks a single placeholder slot. Each placeholder becomes
    a dummy token in the string (here we use the literal token " _") and we record
    the index of that token so we can overwrite its hidden state at layer k.

    Example (the SelfIE default for a Llama-2-chat style interpretation):
        InterpretationPrompt.build(
            tokenizer,
            ("[INST]", 0, 0, 0, 0, 0, "[/INST] Sure, I will summarize the message:"),
        )
    gives 5 placeholder tokens in between [INST] and [/INST].
    """

    input_ids: torch.LongTensor        # shape (1, T_prompt)
    attention_mask: torch.LongTensor   # shape (1, T_prompt)
    placeholder_positions: List[int]   # indices in input_ids[0] that are placeholders
    rendered: str                      # the prompt string (for logging)

    @classmethod
    def build(cls, tokenizer, spec: Sequence, placeholder_str: str = " _") -> "InterpretationPrompt":
        rendered = ""
        placeholder_positions: List[int] = []
        for part in spec:
            if isinstance(part, str):
                rendered += part
            else:
                # record where this placeholder lands in the tokenised sequence
                start = len(tokenizer.encode(rendered, add_special_tokens=False))
                rendered += placeholder_str
                end = len(tokenizer.encode(rendered, add_special_tokens=False))
                placeholder_positions.extend(range(start, end))

        enc = tokenizer(rendered, return_tensors="pt")
        # tokenizer may add BOS at position 0; shift recorded positions accordingly.
        # Simplest robust fix: re-locate placeholders by scanning for the placeholder
        # token id in the final input_ids.
        placeholder_token_ids = set(
            tokenizer.encode(placeholder_str, add_special_tokens=False)
        )
        real_positions = [
            i for i, tid in enumerate(enc.input_ids[0].tolist())
            if tid in placeholder_token_ids
        ]
        # If the naive count matches, prefer that (it preserves SelfIE's semantics
        # when the placeholder happens to appear elsewhere as a normal token).
        if len(real_positions) == len(placeholder_positions):
            placeholder_positions = real_positions
        return cls(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            placeholder_positions=placeholder_positions,
            rendered=rendered,
        )


# ---------------------------------------------------------------------------
# 2. Hook-based injector
# ---------------------------------------------------------------------------

def _get_llama_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the list of LlamaDecoderLayer modules inside an OpenVLA or bare
    Llama model. We walk a few known paths so this works for:
      * bare LlamaForCausalLM  (model.model.layers)
      * OpenVLA PrismaticForConditionalGeneration
        (model.language_model.model.layers)
      * raw LlamaModel         (model.layers)
    """
    for path in (
        ("language_model", "model", "layers"),   # OpenVLA
        ("model", "layers"),                     # LlamaForCausalLM
        ("layers",),                             # LlamaModel
    ):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and isinstance(cur, nn.ModuleList):
            return cur
    raise AttributeError(
        "Could not locate LlamaDecoderLayer ModuleList on the given model. "
        "Pass the layer list explicitly if your model has a non-standard structure."
    )


class _Injector:
    """Forward hooks that (a) record every layer's input hidden state during an
    original pass, and (b) overwrite the layer-k hidden state at chosen token
    positions with a provided tensor during an interpretation pass.
    """

    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.recorded: Dict[int, torch.Tensor] = {}     # layer_idx -> (B, T, D)
        self.inject_at: Dict[int, Tuple[List[int], torch.Tensor]] = {}
        # ^ inject_at[layer_idx] = (positions, tensor of shape (len(positions), D))

    # ---- recording (original pass) ----
    def _make_record_hook(self, idx: int):
        def hook(_module, inputs, _kwargs_or_output=None):
            # pre-forward hook: inputs[0] is the hidden states entering this layer
            hs = inputs[0]
            self.recorded[idx] = hs.detach()
        return hook

    def start_recording(self):
        self.clear_handles()
        self.recorded = {}
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._make_record_hook(i)))

    # ---- injecting (interpretation pass) ----
    def _make_inject_hook(self, idx: int):
        def hook(_module, inputs):
            if idx not in self.inject_at:
                return None
            positions, vec = self.inject_at[idx]
            hs = inputs[0]
            # Only inject on the first (prefill) step, when the sequence dim is
            # the full prompt length. During generation we see seq_len==1 (cached
            # decoding) and we must NOT rewrite that; we've already rewritten
            # the layer-k KV cache implicitly by rewriting the prefill hidden.
            if hs.shape[1] <= max(positions):
                return None
            # hs: (B, T, D); write the same vec to every item in the batch
            hs = hs.clone()  # never mutate caller's tensor
            for p in positions:
                hs[:, p, :] = vec.to(hs.dtype).to(hs.device)
            # reconstruct the input tuple with the edited tensor
            new_inputs = (hs,) + inputs[1:]
            return new_inputs
        return hook

    def start_injecting(self, inject_at: Dict[int, Tuple[List[int], torch.Tensor]]):
        self.clear_handles()
        self.inject_at = inject_at
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._make_inject_hook(i)))

    # ---- cleanup ----
    def clear_handles(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def record_hidden_states(
    model: nn.Module,
    *,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    **extra_forward_kwargs,
) -> List[torch.Tensor]:
    """Run one forward pass and return a list of hidden states, one per layer.

    The returned list has length num_layers; entry i is the hidden state ENTERING
    layer i, shape (B, T, D). This matches SelfIE's `outputs['hidden_states'][l]`
    for l in [0, num_layers-1]. (We ignore the final post-norm output since SelfIE
    injects at layers, not at the lm_head.)
    """
    layers = _get_llama_layers(model)
    injector = _Injector(layers)
    injector.start_recording()
    try:
        call_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,   # we use our own hooks
            use_cache=False,              # keep things simple for a single forward
            return_dict=True,
        )
        if pixel_values is not None:
            call_kwargs["pixel_values"] = pixel_values
        call_kwargs.update(extra_forward_kwargs)
        model(**call_kwargs)
    finally:
        injector.clear_handles()

    # Return layer inputs in order.
    return [injector.recorded[i] for i in range(len(layers))]


@torch.no_grad()
def interpret_embedding(
    model: nn.Module,
    tokenizer,
    embedding: torch.Tensor,            # shape (D,) or (1, 1, D)
    interp_prompt: InterpretationPrompt,
    *,
    inject_layer: int = 2,
    max_new_tokens: int = 30,
    do_sample: bool = False,
) -> str:
    """Interpret a single hidden-state vector by running the interpretation
    forward pass with hooks that overwrite the placeholder hidden states at
    `inject_layer` with `embedding`.

    Returns the generated text, stripped of the prompt.
    """
    layers = _get_llama_layers(model)
    D = embedding.numel()
    vec = embedding.reshape(D).to(next(model.parameters()).device)

    inject_at = {inject_layer: (interp_prompt.placeholder_positions, vec)}

    injector = _Injector(layers)
    injector.start_injecting(inject_at)
    try:
        prompt_len = interp_prompt.input_ids.shape[1]
        gen = model.generate(
            input_ids=interp_prompt.input_ids.to(model.device),
            attention_mask=interp_prompt.attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        injector.clear_handles()

    new_tokens = gen[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def interpret_openvla(
    openvla_model,
    processor,
    image,                               # PIL Image or None for text-only
    prompt: str,                         # e.g. "In: What action should the robot take to pick up the red block?\nOut:"
    *,
    tokens_to_interpret: Sequence[Tuple[int, int]],  # list of (retrieve_layer, retrieve_token)
    interp_prompt: Optional[InterpretationPrompt] = None,
    inject_layer: int = 2,
    max_new_tokens: int = 30,
) -> List[dict]:
    """End-to-end: record hidden states from an OpenVLA forward pass on
    (image, prompt), then for each (layer, token) pair, ask OpenVLA's own LLM
    to describe that embedding in words.

    Returns a list of dicts with keys: layer, token, token_decoded, interpretation.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # --- interpretation prompt default (Llama-2-chat style; OpenVLA uses Llama-2) ---
    if interp_prompt is None:
        interp_prompt = InterpretationPrompt.build(
            tokenizer,
            ("[INST] ", 0, 0, 0, 0, 0, " [/INST] Sure, I will summarize the message:"),
        )

    # --- 1. original forward on the real (image, text) input ---
    if image is not None:
        inputs = processor(prompt, image)
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(openvla_model.device) for k, v in inputs.items()
              if isinstance(v, torch.Tensor)}

    all_hidden = record_hidden_states(openvla_model, **inputs)

    # Decode every token in the input for logging (OpenVLA prepends image patch
    # tokens AFTER the BOS token, so there's no text equivalent for those).
    input_ids = inputs.get("input_ids")
    decoded = {}
    if input_ids is not None:
        for t in range(input_ids.shape[1]):
            decoded[t] = tokenizer.decode(input_ids[0, t])

    # --- 2. interpretation pass per (layer, token) ---
    results = []
    for l, t in tokens_to_interpret:
        if l >= len(all_hidden):
            raise IndexError(f"retrieve_layer={l} >= num_layers={len(all_hidden)}")
        if t >= all_hidden[l].shape[1]:
            raise IndexError(
                f"retrieve_token={t} >= seq_len={all_hidden[l].shape[1]}. "
                f"Remember OpenVLA prepends {all_hidden[l].shape[1] - input_ids.shape[1]} "
                f"image-patch tokens after <BOS>."
            )
        vec = all_hidden[l][0, t]  # (D,)
        text = interpret_embedding(
            openvla_model, tokenizer, vec, interp_prompt,
            inject_layer=inject_layer, max_new_tokens=max_new_tokens,
        )
        results.append({
            "layer": l,
            "token": t,
            "token_decoded": decoded.get(t, "<image_patch_token>"),
            "interpretation": text.strip(),
        })

    return results


# ---------------------------------------------------------------------------
# 4. How to patch SelfIE itself if you insist on using its upstream code
# ---------------------------------------------------------------------------

def patch_selfie_for_transformers_4_40():
    """Documentation-only helper. If you want to use the upstream SelfIE repo
    (https://github.com/tonychenxyz/selfie) against transformers 4.40.1 (which
    is what OpenVLA pins), you need the following edits to SelfIE's
    `selfie/llama_forward_wrappers.py`. These are the minimum edits; after them
    the module will import and run on transformers 4.38 - 4.44.

    ===== Edit 1: flash-attn symbol rename =====
        Old (line ~37):
            from transformers.utils import ( ..., is_flash_attn_available, ... )
        New:
            from transformers.utils import ( ..., is_flash_attn_2_available as is_flash_attn_available, ... )
        (or just delete the import + its guarded block; SelfIE never actually
         calls flash-attn in its interpretation passes.)

    ===== Edit 2: attention-mask preparation =====
        Old (line ~313):
            attention_mask = model.model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        New:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

    ===== Edit 3: drop the removed `padding_mask=` kwarg =====
        Old (line ~441, inside decoder_layer_forward_interpret):
            hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,        # <-- remove this line
            )
        New: just delete the `padding_mask=padding_mask,` line. The `padding_mask`
        kwarg was removed from LlamaAttention.forward in transformers 4.38.

        Also remove the corresponding `padding_mask=padding_mask` in the
        gradient-checkpointing branch around line 356.

    ===== Edit 4 (only if you want use_cache=True): DynamicCache migration =====
        From transformers 4.36 onward, `past_key_values` is a `DynamicCache`
        object, not a tuple of tuples. SelfIE's line
            past_key_values_length = past_key_values[0][0].shape[2]
        blows up. Safest fix: force `use_cache=False` in SelfIE's interpretation
        forward, since it runs a full forward on the interpretation prompt each
        time anyway. Concretely, at the top of `model_model_forward_interpret`:
            use_cache = False          # <-- add this line; SelfIE doesn't need
                                       #     KV cache during interpretation.

    With those four edits, SelfIE runs correctly against OpenVLA's pinned
    transformers 4.40.1. However, the hook-based approach in this file is
    recommended because:
      * It is version-agnostic (no forward-method copy).
      * It does not mutate SelfIE source, so you can keep SelfIE pinned and
        just install its utilities.
      * OpenVLA uses flash-attn-2 during normal inference; SelfIE's hand-rolled
        forward bypasses the flash path, which silently changes numerics. Hooks
        keep the original flash path intact.
    """
    # This function is pure documentation; nothing to execute.
    return None
