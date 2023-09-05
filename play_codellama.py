# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama
import sys

def getRole ():
    print (">>[role]:")
    for line in sys.stdin:
        role = line.rstrip()
        break
    if not role in ["user", "system"]:
        return "user"
    return role

def getContent ():
    print (">>[content]:")
    for line in sys.stdin:
        content = line.rstrip()
        return content
    
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    while True:
        instructions = []

        role = getRole ()
        content = getContent ()
        instructions.append ({"role":role, "content":content})

        results = generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for instruction, result in zip(instructions, results):
            for msg in instruction:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
