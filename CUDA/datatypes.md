# CUDA Data Types & Best Practices Cheatsheet

## 1. Standard Scalars (The Baseline)
*Use these for general logic and standard arithmetic.*

| Type | Size (Bytes) | Description | Best Practice |
| :--- | :--- | :--- | :--- |
| **`float`** | 4 | Standard FP32 | **Default for math.** Fast (1x throughput). |
| **`int`** | 4 | Signed Integer | Loop counters & general logic. |
| **`size_t`** | 8 (64-bit) | Unsigned Size | **MANDATORY for array indexing** to prevent overflow on large data. |
| **`double`** | 8 | FP64 | **Slow on consumer/T4 GPUs** (1/32 speed). Use only for high-precision physics. |
| **`bool`** | 1 | Boolean | Warning: Occupies 1 byte (8 bits), not 1 bit. |
| **`char`** | 1 | Byte | Raw byte manipulation / quantization. |

---

## 2. Built-in Vector Types (The Bandwidth Optimizers)
*Use these to saturate memory bandwidth. Aligned to hardware registers.*
*Pattern: `type` + `1, 2, 3, 4` (e.g., `float4`)*

| Type | Max Size | Best Use Case |
| :--- | :--- | :--- |
| **`float4`** | 16B (128-bit) | **High Bandwidth Memory Access.** Loads 4 floats in 1 instruction. |
| **`int4`** | 16B (128-bit) | Moving integer arrays efficiently. |
| **`uchar4`** | 4B (32-bit) | **RGBA Images.** Represents one pixel perfectly. |
| **`double2`**| 16B (128-bit) | Efficient `double` loading. |

> **⚠️ THE TRAP:** Avoid `type3` (e.g., `float3`, `uchar3`) for arrays. They break 128-byte memory coalescing alignment. Use `type4` with padding instead.

### 🧠 Why the limit? (Hardware Insight)
These are not just software structs; they map directly to hardware instructions. Your T4 GPU has specific assembly (PTX) instructions like:

* `LD.E.32` (Load 32 bits / 1 int)
* `LD.E.64` (Load 64 bits / int2)
* `LD.E.128` (Load 128 bits / int4)

There is **no hardware instruction** to "Load 56 integers" at once. The maximum a single thread can load in one standard instruction is **128 bits (16 bytes)**.

* **float4** = $4 \times 32 \text{ bits} = 128 \text{ bits}$ (Max Hardware Limit)
* **int4** = $4 \times 32 \text{ bits} = 128 \text{ bits}$ (Max Hardware Limit)
* **double2** = $2 \times 64 \text{ bits} = 128 \text{ bits}$ (Max Hardware Limit)

---

## 3. AI & Low Precision (Tensor Core Types)
*Requires specific headers. Trades precision for massive speedups (2x-8x).*

| Type | Header | Description | T4/Turing Support |
| :--- | :--- | :--- | :--- |
| **`__half`** | `<cuda_fp16.h>` | **FP16.** Deep Learning Inference. | ✅ Native 2x Speed |
| **`__half2`** | `<cuda_fp16.h>` | **Vectorized FP16.** Packs 2 halves in 1 reg. | ✅ Native 2x Speed |
| **`__nv_bfloat16`** | `<cuda_bf16.h>` | **BF16.** Training stability (wide range). | ✅ Supported |
| **`nvcuda::wmma`** | `<mma.h>` | **Matrix Fragments.** Used for Tensor Cores. | ✅ Supported |

---

## 4. Host & Grid Management Types
*Used for defining kernel execution parameters.*

| Type | Description | Example |
| :--- | :--- | :--- |
| **`dim3`** | 3D Vector (x,y,z) for Grid/Block dimensions. | `dim3 threads(256);` |
| **`cudaStream_t`** | Execution queue handle (for async concurrency). | `cudaStreamCreate(&stream);` |
| **`cudaEvent_t`** | Timing and synchronization marker. | `cudaEventRecord(start);` |

---

## 5. Quick Selection Guide (Decision Tree)

1.  **Calculating General Math?**
    * ✅ Use `float`
2.  **Moving Data from RAM to Registers?**
    * ✅ Cast to `float4` (or `int4`)
3.  **Indexing Arrays / Loops?**
    * ✅ Use `size_t` (Avoid `int` overflow)
4.  **Deep Learning Inference?**
    * ✅ Use `__half` (FP16)
5.  **Image Processing?**
    * ✅ Use `uchar4` (RGBA) or cast `uchar3` manually.
6.  **Need Scientific Precision?**
    * ⚠️ Use `double` (Accept the speed penalty)