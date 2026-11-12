# llama-rust

A small LLM inference engine written from scratch in Rust, in the spirit of [`llama.cpp`](https://github.com/ggml-org/llama.cpp). The goal is to load a quantized GGUF model and generate text with no ML framework, just `std` and one crate for memory mapping.

> **Status: early work in progress.** Milestone 1 of 7 is done: the GGUF loader reads a real 4.5 GB model. It can't generate text yet.

## What works today

Point it at a GGUF file and it:

- **Memory-maps** the model instead of reading it into RAM, so the 4.5 GB file is available without being copied.
- **Checks the `GGUF` magic bytes** and reads the version, tensor count and metadata count from the header.
- **Parses all metadata** key/value pairs. This covers all 13 GGUF value types, including nested arrays such as the 248,320-entry vocabulary.
- **Reads the tensor directory** for every tensor (name, shape, quantization type, offset) into a `HashMap<&str, TensorInfo>`.
- **Finds the start of the raw weights** by rounding the directory's end up to the next 32-byte boundary.

Strings and tensor names borrow straight from the mapped file (`&'a str`) instead of being copied.

Output on the model it was built against:

```text
GGUF Version: 3
Tensor count: 426
Metadata KV count: 33
0:general.architecture = String("qwen35")
...
25: tokenizer.ggml.tokens= Array of 248320 items
27: tokenizer.ggml.merges= Array of 247587 items
0:'output_norm.weight' shape=[2560] type=0 offset=0
1:'token_embd.weight' shape=[2560, 248320] type=8 offset=10240
2:'blk.0.attn_gate.weight' shape=[2560, 4096] type=8 offset=675440640
...
Successfull loaded 426 tensor into map!
Raw tensor data start at byte:10967392
```

## Quick start

You need a recent stable Rust toolchain. The crate uses edition 2024, so Rust 1.85 or newer.

```bash
cargo run --release -- /path/to/model.gguf
```

With no argument it looks for `models/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf` inside the repo. `models/` and `*.gguf` are git-ignored, so you can drop a model there without committing it.

It has only been run against the Qwen3.5 model below. The header and metadata parsing is generic GGUF, but other models are untested and an unknown value type currently panics.

## How the GGUF file is read

```text
┌──────────────────────────────┐
│ magic "GGUF"        4 bytes  │
│ version             u32      │
│ tensor_count        u64      │
│ metadata_kv_count   u64      │
├──────────────────────────────┤
│ metadata: key, type, value   │  × metadata_kv_count
├──────────────────────────────┤
│ tensor info: name, n_dims,   │  × tensor_count
│   shape[], type, offset      │
├──────────────────────────────┤
│ padding to 32-byte boundary  │  (offset + 31) & !31
├──────────────────────────────┤
│ raw tensor data              │  ← what inference will read
└──────────────────────────────┘
```

`ByteReader` walks the mapped bytes with a cursor. It decodes little-endian integers, floats and length-prefixed strings, and `read_value` recurses to handle arrays.

## Target model

| | |
|---|---|
| File | `Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf` (about 4.5 GB) |
| Architecture | `qwen35` |
| Layers | 32 transformer blocks |
| Hidden size / FFN size | 2,560 / 9,216 |
| Attention | 16 query heads, 4 KV heads (grouped-query, 4:1) |
| Context length | 262,144 tokens |
| RoPE base frequency | 10,000,000 |
| Vocabulary | 248,320 tokens, byte-level BPE |
| Tensors | 426 (Q8_0 weights, F32 norm layers) |

## Roadmap

```text
[1 Loader]  →  [2 Q8_0 dequant]  →  [3 Math kernels]  →  [4 Transformer block]
                                                                  ↓
          [7 Optimize]  ←  [6 Generation loop]  ←  [5 Tokenizer]
```

- [x] **1. GGUF loader**: mmap, header, metadata, tensor directory, data offset
- [ ] **2. Weight slicing and Q8_0**: slice any tensor out of the map, read F32 layers, dequantize Q8_0 blocks (2-byte `f16` scale + 32 `i8` values, `w = d · q`)
- [ ] **3. Core math**: embedding lookup, RMSNorm, GEMV, RoPE, SwiGLU, softmax
- [ ] **4. Transformer block**: grouped-query attention, KV cache, FFN, residuals, LM head
- [ ] **5. Tokenizer**: read the vocab and merges from metadata, byte-level BPE encode and decode
- [ ] **6. Generation loop**: greedy, temperature, top-k and top-p sampling, stop at EOS
- [ ] **7. Performance**: NEON SIMD, multi-threaded GEMV, possibly Accelerate or Metal

## Project layout

```text
llama-rust/
├── Cargo.toml     one dependency: memmap2
├── src/main.rs    GgufValue, TensorInfo, ByteReader, and the loader in main()
└── models/        put your .gguf here (git-ignored)
```

## Acknowledgements

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and the [GGUF format spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) are the reference for everything here.

## License

[MIT](LICENSE)
