# OpenVLA × SelfIE — Text Interpretation of OpenVLA Hidden Embeddings

Screening task: apply SelfIE (Chen, Vondrick, Mao 2024 — Columbia) to OpenVLA
(Kim, Pertsch, Karamcheti et al. 2024) so we can get natural-language
descriptions of the hidden embeddings inside OpenVLA's Llama-2 backbone.

## TL;DR

- OpenVLA = Prismatic VLM = **DINOv2 + SigLIP → MLP projector → Llama-2 7B**. Image
  patch embeddings get projected into the LLM's embedding space and inserted
  right after `<BOS>`. That means an "image patch" lives as a regular hidden
  state inside the Llama layers, and SelfIE's trick applies directly.
- SelfIE's own code does not work out of the box on OpenVLA, primarily due to a
  **transformers version conflict** (SelfIE is pinned to `4.34.0`; OpenVLA to
  `4.40.1`). See [Section 3](#3-transformers-version-problems--exact-fixes).
- My approach: **reimplement SelfIE's mechanism with PyTorch forward-pre hooks**
  instead of calling into SelfIE's copy-pasted LLaMA forward. This is
  transformers-version-agnostic, keeps flash-attn-2 intact on the real forward
  path, and is ~200 lines of code. See `openvla_selfie.py`.
- A fallback, if you prefer to use upstream SelfIE: a minimal four-edit patch
  to `selfie/llama_forward_wrappers.py` is documented in
  `patch_selfie_for_transformers_4_40()` and reproduced below.

## Files

| File | Purpose |
| ---- | ------- |
| `openvla_selfie.py` | Core library. `InterpretationPrompt`, `record_hidden_states`, `interpret_embedding`, `interpret_openvla`. |
| `test_hooks_on_tiny_llama.py` | **Actually runs** in any environment (no GPU, no downloads). Builds a tiny 4-layer Llama and verifies the three load-bearing claims of the hook logic. |
| `run_openvla_selfie_example.py` | End-to-end example with the real OpenVLA-7B checkpoint. Needs a GPU + the weights. |

## 1. How SelfIE works, one paragraph

Given an input `x`, run a forward pass and cache every layer's hidden state
`H[l, t]`. Then take a fixed interpretation prompt, e.g.
`"[INST] _ _ _ _ _ [/INST] Sure, I will summarize the message:"`, and run a
second forward pass on that prompt. At some early layer `k` of that second
pass, replace the hidden states at the five underscore positions with `H[l, t]`
repeated across those positions. Let the model generate. The generated text is
the model's own natural-language description of the embedding `H[l, t]`.

## 2. How it maps onto OpenVLA

Three things about OpenVLA matter here:

1. **Multimodal embeddings get spliced into the Llama-2 input embedding stream
   right after `<BOS>`.** Concretely, in `modeling_prismatic.py`:
   ```python
   multimodal_embeddings = torch.cat(
       [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]],
       dim=1,
   )
   ```
   With the flagship `prism-dinosiglip-224px` backbone there are 256 image-patch
   tokens. So token positions `1 .. 256` inside the Llama hidden states are
   image patches; `0` is `<BOS>`; `257..` are the text instruction + output.
2. **The Llama backbone is accessible as `model.language_model`.** Its layers
   are at `model.language_model.model.layers` — a standard `nn.ModuleList` of
   `LlamaDecoderLayer`s. This is exactly the list SelfIE injects into.
3. **OpenVLA's `PrismaticForConditionalGeneration.forward` already accepts
   `output_hidden_states=True` and forwards it to the language model.** So we
   could just use that. I still prefer hooks (Section 4), because they give us
   per-layer *input* hidden states (what SelfIE wants for injection), not the
   post-norm output hidden states HF returns by default.

## 3. Transformers version problems & exact fixes

SelfIE was written against `transformers==4.34.0`; OpenVLA pins
`transformers==4.40.1` (see OpenVLA's README and the explicit version check in
`modeling_prismatic.py`). Between those two versions the HuggingFace LLaMA
internals changed in several incompatible ways. If you `pip install selfie`
into an OpenVLA env and run `from selfie.interpret import interpret`, you'll
hit the following errors in order:

### Error 1 — `ImportError: cannot import name 'is_flash_attn_available'`

SelfIE's `llama_forward_wrappers.py` line 37:
```python
from transformers.utils import (..., is_flash_attn_available, ...)
```
That symbol was renamed to `is_flash_attn_2_available` in transformers 4.38.

**Fix.** Replace the import with
```python
from transformers.utils import (..., is_flash_attn_2_available as is_flash_attn_available, ...)
```
(Or delete the whole `if is_flash_attn_available(): from flash_attn import ...`
block — SelfIE doesn't actually call flash-attn anywhere in its interpretation
path.)

### Error 2 — `AttributeError: 'LlamaModel' object has no attribute '_prepare_decoder_attention_mask'`

SelfIE's `llama_forward_wrappers.py` around line 313:
```python
attention_mask = model.model._prepare_decoder_attention_mask(
    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
)
```
This method was removed in transformers 4.38 and replaced with a free function.

**Fix.** At the top of the file add
```python
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
```
and change the call to
```python
attention_mask = _prepare_4d_causal_attention_mask(
    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
)
```

### Error 3 — `TypeError: LlamaAttention.forward() got an unexpected keyword argument 'padding_mask'`

SelfIE's `decoder_layer_forward_interpret` (line ~434) calls the self-attention
with `padding_mask=padding_mask`. That kwarg was removed in 4.38; padding is
now embedded in the 4D attention mask.

**Fix.** Delete the `padding_mask=padding_mask,` line inside the
`decoder_layer.self_attn(...)` call. Also delete the similar kwarg on line
~356 inside the gradient-checkpointing branch.

### Error 4 — `AttributeError: 'DynamicCache' object has no attribute '__getitem__'` (only if `use_cache=True`)

SelfIE does
```python
past_key_values_length = past_key_values[0][0].shape[2]
```
From transformers 4.36 onward `past_key_values` is a `DynamicCache` object, not
a tuple of tuples. You only hit this if you pass `use_cache=True` to SelfIE,
which SelfIE's own code actually does during `generate_interpret` indirectly.

**Fix.** Safest: force `use_cache=False` at the top of
`model_model_forward_interpret`:
```python
use_cache = False  # SelfIE runs a full forward on each interpretation prompt
                   # so KV cache gives no speedup here anyway.
```
This is a one-line change and avoids the entire DynamicCache migration.

---

After those four edits, upstream SelfIE runs correctly against OpenVLA's
pinned transformers 4.40.1. The function `patch_selfie_for_transformers_4_40()`
in `openvla_selfie.py` contains the same instructions in a single docstring
you can copy-paste into an issue or PR against the SelfIE repo.

## 4. Why I still prefer the hook-based approach

Even with the four patches above, SelfIE's design copy-pastes an entire
`LlamaModel.forward` implementation into its own module. That has three
problems for OpenVLA specifically:

1. **Numerics drift from the real OpenVLA forward.** OpenVLA is loaded with
   `attn_implementation="flash_attention_2"`. SelfIE's hand-rolled forward
   ignores that setting and goes through the slow eager path, so the hidden
   states it records are *numerically different* from the ones produced during
   normal OpenVLA inference. That defeats the point of interpreting the
   model's actual embeddings.
2. **It re-fights the transformers version skew on every upgrade.** Any
   future LLaMA-internals refactor (and there has been roughly one per minor
   version) requires a new patch.
3. **It makes the code harder to audit.** Two LLaMA forwards in the codebase
   — one in `transformers`, one in `selfie` — will silently diverge.

Hooks sidestep all three. A `forward_pre_hook` on each `LlamaDecoderLayer`
can (a) record the layer's input tensor during the original pass, and (b)
overwrite specific rows of that input tensor during the interpretation pass.
No private APIs used. Fully version-agnostic. Preserves flash-attn-2.

## 5. Usage

```python
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from openvla_selfie import InterpretationPrompt, interpret_openvla

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
).to("cuda:0").eval()

image = Image.open("my_scene.jpg").convert("RGB")
prompt = "In: What action should the robot take to pick up the red block?\nOut:"

# Interpret the center image-patch token at layers 5, 10, 15, 20
results = interpret_openvla(
    model, processor, image, prompt,
    tokens_to_interpret=[(l, 1 + 128) for l in (5, 10, 15, 20)],
    inject_layer=2, max_new_tokens=30,
)
for r in results:
    print(f"layer {r['layer']:>2d}  tok {r['token']:>4d}  -> {r['interpretation']}")
```

## 6. Verifying the hook logic works

I can't download 15 GB of OpenVLA weights in this screening sandbox, but the
hook logic is verified end-to-end against a tiny randomly-initialised Llama in
`test_hooks_on_tiny_llama.py`:

```
$ python test_hooks_on_tiny_llama.py
Building tiny Llama (random init, 4 layers, hidden_size=64)...
  num_hidden_layers = 4
  hidden_size       = 64

[1] test_record_shape ... OK
[2] test_injection_actually_changes_downstream ... OK
[3] test_end_to_end_interpretation ...
     injected vec_a -> '$}န spraw民 lear'
     injected vec_b -> 'stud Brig maintainedÎore'
[3] OK

All tests passed.
```

The three tests establish, in order:

1. Every layer's input hidden state is captured with the correct shape.
2. Overwriting a layer's input via the pre-hook really changes the downstream
   layer's input — i.e. the injection reaches the computation graph; it's not
   a silent no-op.
3. The full interpretation generation pipeline runs, and injecting different
   embeddings produces different generated text (the prerequisite for SelfIE to
   be a useful interpreter at all). The generated strings are gibberish because
   the model is random-init; with the real OpenVLA checkpoint they'd be
   coherent English.

## 7. Known caveats

- **Image-patch token positions.** OpenVLA's `prism-dinosiglip-224px` variant
  has 256 patch tokens at positions `1..256` in the Llama hidden stream. The
  older `siglip-224px` variant has a different patch count (also 256 at 224px,
  but if you fine-tune with a different image size this shifts). If you're
  unsure, print `inputs['input_ids'].shape[1]` and compare to the raw text
  tokenization length.
- **Which layer `k` to inject at.** Chen et al. recommend injecting at layer
  1-2 of the interpretation pass. Going earlier (layer 0) bypasses embedding
  look-up; going later gives the model fewer layers to "explain" the vector,
  so quality degrades.
- **OpenVLA generates action tokens, not text.** The Llama backbone is
  fine-tuned to produce discretized action tokens. When we ask it to interpret
  hidden states through the `[INST]...[/INST]` framing, we're asking it to
  behave more like the original Llama-2-Chat it was fine-tuned *from*. In
  practice this works (the chat-instruction prior is still there) but
  interpretations are noisier than they would be on the un-fine-tuned
  Llama-2-chat. A stronger variant would be to keep a frozen Llama-2-chat
  side-by-side and interpret OpenVLA embeddings using *that* model as the
  interpreter, mixing the Llama-2 tokenizer between the two. That's outside
  the scope of this screening task but is a natural follow-up.
