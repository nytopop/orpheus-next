import argparse
import asyncio

import asyncstdlib as A
import torch
import torchaudio
from openai import AsyncOpenAI

from orpheus_next import decode_orpheus


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api-key", metavar="OPENAI_API_KEY", default="foobar")
    parser.add_argument("-u", "--api-url", metavar="OPENAI_API_URL", default="http://localhost:8000/v1")
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_url)

    # if using a variant without any chat template; base completions api works as well
    gen = await client.chat.completions.create(
        model="orpheus",
        messages=[{"role": "user", "content": args.text}],
        stream=True,
        temperature=0.7,
        top_p=0.9,
        max_completion_tokens=5000,
    )

    gen = A.map(lambda c: c.choices[0].delta.content, gen)
    gen = A.filter(lambda t: t is not None, gen)

    audio = torch.cat([au async for au in decode_orpheus(gen)], dim=1).cpu()

    torchaudio.save("out.wav", audio, sample_rate=24000, channels_first=True)


if __name__ == "__main__":
    asyncio.run(main())
