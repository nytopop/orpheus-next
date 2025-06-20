# orpheus-next
Experimental work into TTS based on [Orpheus by Canopy Labs](https://huggingface.co/canopylabs) (no affiliation).

This repo contains:
- a library to use orpheus style models via any inference provider (vLLM, sglang, llama.cpp, etc)
- finetuning scripts

## library features
- [x] improved time-to-first-frame latency: 3 x 7 token sliding windows instead of 4 x 7 (+ it's configurable)
- [x] optional non-streaming decoder: significantly less overhead if you don't need streaming responses
- [x] significantly more robust decoder logic that doesn't break upon imperfect model output
- [x] supports arbitrary inference providers: bring your own tokens

## finetuned model features (soon.jpg)
- [x] conversational context adherence
- [x] speaker control tags
- [x] style control tags
- [ ] paralinguistic control tags

## usage

```python
from openai import OpenAI
from orpheus_next import decode_orpheus
import torch
import torchaudio

client = OpenAI(api_key="...", base_url="...")

gen = client.chat.completions.create(
    model="orpheus",
    messages=[{"role": "user", "content": "I'm a speech generation model that sounds like a person."}],
    stream=True,
    temperature=0.7,
    max_completion_tokens=5000,
)

gen = map(lambda c: c.choices[0].delta.content, gen)

audio = torch.cat([frame for frame in decode_orpheus(gen)], dim=1).cpu()

torchaudio.save("out.wav", audio, sample_rate=24000, channels_first=True)
```
