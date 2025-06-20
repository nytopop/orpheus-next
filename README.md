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
TODO
