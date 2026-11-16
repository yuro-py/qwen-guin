# qwen-guin

just run:-

```
python3 kv-quantization-triton-paged.py
```
for the overall output.

You can change:-
1. prompt in "PROMPT"
2. prompt length in "MAX_NEW_TOKENS" according to your gpu vram
3. set a custom block size in "BLOCK_SIZE"
4. change sampling parameters under model declaration
5. u can turn off quantization in "KV_QUANT_INT8 = True" with False.
