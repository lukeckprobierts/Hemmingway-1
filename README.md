# Hemmingway-1

A 27B model that writes the way a person writes.

Hemmingway-1 is built by [Altworld](https://hemmingway.io) on top of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It is trained for two
things: writing that does not read as machine-written, and answering a person's
whole message instead of a piece of it. Weights are Apache-2.0.

- **Weights:** [Altworld/Hemmingway-1](https://huggingface.co/Altworld/Hemmingway-1)
- **Try it:** [hemmingway.io](https://hemmingway.io)
- **Apps:** [hemmingway.io/download](https://hemmingway.io/download) (Mac, Android)

## What it is

| | |
|---|---|
| Parameters | 27B |
| Base | Qwen/Qwen3.8-27B |
| Context | 262,144 tokens |
| Licence | Apache-2.0 |
| Model id | `hemmingway-27b` |

## How it scores

Our own benchmark. Every reply is judged against another model's reply to the
same prompt, blind, in both orders, and the ratings are Bradley-Terry. The judge
is GLM-5.3 at low thinking.

| model | Elo |
|---|---:|
| **Hemmingway-1** | **1197** |
| Kimi K3 | 1197 |
| Qwen3.8-Max | 1081 |
| DeepSeek V4 Pro | 1054 |
| DeepSeek V4 Flash | 981 |
| Gemma 4 31B | 808 |
| Qwen3.8-27B (base) | 693 |

Level with Kimi K3, and 504 points above the base model it was trained from.
Closed frontier models still score above it.

This is our own benchmark, run by us. Take it as ours, not as a neutral result.

## Running it

vLLM:

```bash
vllm serve Altworld/Hemmingway-1 --max-model-len 262144
```

Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Altworld/Hemmingway-1"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")

messages = [{"role": "user", "content": "Write the text I send my landlord about the broken boiler."}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=512)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

## Limits

English first. It can be wrong and still sound certain. Not for medical, legal
or financial decisions.

## Licence

Apache-2.0, the same as the base model. See [LICENSE](LICENSE).
