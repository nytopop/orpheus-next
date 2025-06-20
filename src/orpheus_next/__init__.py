import itertools as I
import logging
from typing import AsyncIterator, Iterator, List

import torchaudio.functional as F
import asyncstdlib as A
import torch
from snac import SNAC

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self):
        self.buffer = ""

    def feed(self, input: str):
        """Feed text to the parser, yielding (cb, code) tuples as they become available."""

        self.buffer += input

        while True:
            # find the next token
            start = self.buffer.find("<custom_token_")
            if start == -1:
                break  # nothing 2 do

            end = self.buffer.find(">", start)
            if end == -1:
                break  # incomplete

            end = end + 1

            # extract / validate
            token = self.buffer[start:end]
            self.buffer = self.buffer[end:]

            try:
                id = int(token[14:-1])
            except ValueError:
                continue

            if 10 <= id <= 28682:
                id = id - 10
                cb = id // 4096
                yield (cb, id - (cb * 4096))


_codec_device = "cuda" if torch.cuda.is_available() else "cpu"

_codec = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(_codec_device)

_codec.decode(
    [  # warmup
        torch.tensor([[5] * 4]).to(_codec_device),
        torch.tensor([[5] * 8]).to(_codec_device),
        torch.tensor([[5] * 16]).to(_codec_device),
    ]
)


def _decode_orpheus(codes: List[int]) -> torch.Tensor:
    """Decode a list of SNAC codes in orpheus layout as audio."""

    assert len(codes) % 7 == 0 and len(codes) >= 7
    assert all(0 <= c < 4096 for c in codes)

    _12 = [[c for i, c in enumerate(codes) if i % 7 == 0]]
    _24 = [[c for i, c in enumerate(codes) if i % 7 in (1, 4)]]
    _48 = [[c for i, c in enumerate(codes) if i % 7 in (2, 3, 5, 6)]]

    _12 = torch.tensor(_12, dtype=torch.int32, device=_codec_device)
    _24 = torch.tensor(_24, dtype=torch.int32, device=_codec_device)
    _48 = torch.tensor(_48, dtype=torch.int32, device=_codec_device)

    with torch.inference_mode():
        return _codec.decode([_12, _24, _48]).squeeze(0)


def _decode_orpheus_sw(mframe: List[int]) -> torch.Tensor:
    """Decode one frame of audio (85.333ms) from a sliding window of orpheus SNAC frames."""

    assert 3 <= len(mframe) // 7

    return _decode_orpheus(mframe)[:, 2048:4096]


class Decoder:
    def __init__(self, window: int = 3):
        self.expecting = 0
        self.codes = []
        self.parser = Parser()

        if window >= 3:
            self.window = window * 7
        else:
            raise ValueError(f"window must be at least 3, but got {window}")

    def feed_sw(self, input: str):
        """Feed text into the decoder, yielding decoded audio frames."""

        for frame in self.feed(input):
            yield _decode_orpheus_sw(frame)

    def feed(self, input: str):
        """Feed text into the decoder, yielding SNAC frames."""

        for cb, code in self.parser.feed(input):
            if cb != self.expecting:
                logger.debug(f"skipping (cb={cb}, code={code})' because expected cb={self.expecting}")
                self.codes = self.codes[: -self.expecting] if self.expecting else self.codes
                self.expecting = 0
                continue

            self.expecting = (self.expecting + 1) % 7
            self.codes.append(code)

            if len(self.codes) % 7 != 0 or len(self.codes) < self.window:
                continue

            yield self.codes[-self.window :]


def decode_orpheus(input: str | Iterator[str] | AsyncIterator[str], window: int = 3):
    """Decode orpheus tokens to audio tensor(s).

    Args:
        input: Orpheus tokens as string, iterator, or async iterator of string chunks.
        window: Sliding window size for streaming (minimum 3 ~85.34ms frames).

    Returns:
        For string input: Complete decoded audio tensor (on codec device).
        For streaming input: Iterator/async iterator of audio chunks (on codec device).
    """

    if isinstance(input, str):  # higher throughput non-streaming impl
        codes = list(I.chain.from_iterable(Decoder(window=1).feed(input)))
        return _decode_orpheus(codes)

    dec = Decoder(window=window)

    if hasattr(input, "__aiter__"):
        return A.chain.from_iterable(A.map(dec.feed_sw, input))

    if hasattr(input, "__iter__"):
        return I.chain.from_iterable(map(dec.feed_sw, input))

    raise TypeError(f"unsupported input type: {type(input)}")


def encode_orpheus(audio: torch.Tensor, sr: int = 24000) -> str:
    """Encode audio tensor to orpheus tokens.

    Args:
        audio: Audio tensor of shape [T] or [C, T]. If C > 1, channels are averaged to mono.
        sr: Sample rate of input audio

    Returns:
        String of orpheus tokens in format "<custom_token_N><custom_token_M>..."
    """

    if audio.dim() not in (1, 2):
        raise ValueError(f"expected audio of shape [T] or [C, T] but got {audio.shape}")

    audio = audio.to(device=_codec_device)

    if sr != 24000:
        audio = F.resample(audio, orig_freq=sr, new_freq=24000)

    if audio.dim() == 2:
        audio = audio.mean(dim=0)

    # SNAC time
    with torch.inference_mode():
        [_12, _24, _48] = _codec.encode(audio[:, None, None])

    _12 = _12.squeeze(0).reshape(-1, 1)  # [N, 1], 12hz discrete codes
    _24 = _24.squeeze(0).reshape(-1, 2)  # [N, 2], 24hz discrete codes
    _48 = _48.squeeze(0).reshape(-1, 4)  # [N, 4], 48hz discrete codes

    # orpheus frame layout [12, 24, 48, 48, 24, 48, 48]
    codes = torch.cat([_12, _24[:, :1], _48[:, :2], _24[:, 1:], _48[:, 2:]], dim=1)

    # inductive bias; each position is drawn from an independent codebook
    codes = codes + torch.arange(start=0, end=28672, step=4096, device=codes.device) + 10
    codes = codes.flatten().tolist()

    return "".join(f"<custom_token_{code}>" for code in codes)
