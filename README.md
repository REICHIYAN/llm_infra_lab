# 🚀 LLM-Infra-Lab
A minimal, reproducible LLM infrastructure stack — designed so anyone can understand the *entire LLM pipeline* in under an hour.

This project is for engineers who want to learn:

- How fast inference engines (like vLLM) organize KV caches & batching  
- How distributed training (FSDP, JAX pmap) is wired internally  
- How serving layers + infra (K8s / Terraform) fit around an LLM  
- How to test the whole thing on a *single CPU-only laptop*

No GPUs required. No massive models.  
Just clean, readable, production-shaped code — the smallest possible LLM infra stack you can actually learn from.

> 🧠 **If full LLM systems feel like a black box, this repo makes them transparent.**

---

## ⭐ Why this exists

Most LLM repos are either:

- too huge to read, or  
- too toy-like and unrealistic.

This lab sits in the middle:  
**small enough to understand structurally, but real enough to teach the architecture correctly.**

You get:

- A working KV-cache engine  
- A working FastAPI inference server  
- A working FSDP-style training step  
- A working JAX pmap example  
- A working K8s/Terraform infra skeleton  
- A complete pytest suite verifying everything

All runnable on **CPU only**.  
Everything is **minimal**, but nothing is **fake**.

---

# 🔥 Try it in 10 seconds (verified on Colab & CPU-only)

```bash
git clone https://github.com/REICHIYAN/llm_infra_lab.git
cd llm_infra_lab
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

If you see:

```
5 passed in X.XXs
```

You now have:

- working inference server  
- working KV cache  
- working batching  
- working FSDP fallback  
- working JAX pmap check  

→ **The whole LLM infra stack is verified.**

---

# 🧩 What’s inside (architecture overview)

```
llm_infra_lab/
├── serving/          # FastAPI server + vLLM-style KV cache engine
├── training/         # FSDP-style tiny training loop (works on CPU)
├── jax/              # pmap mini example for distributed concepts
├── tests/            # Full pytest suite (API, KV cache, training)
├── k8s/              # Deployment + Service (GKE-ready)
├── terraform/        # Minimal GKE IaC skeleton (validate without apply)
└── scripts/          # run_server.sh / run_tests.sh
```

---

# ⚡ 1. Run the inference server

```bash
./scripts/run_server.sh
```

Then hit:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate      -H "Content-Type: application/json"      -d '{"prompt": "Hello!", "max_new_tokens": 32}'
```

This triggers:

- `serving/api_server.py`
- `serving/vllm_mock.py` (fast path, KV reuse, batching)

---

# ⚡ 2. Verify everything end-to-end (pytest)

```bash
PYTHONPATH=. pytest -q
```

Includes tests for:

- FastAPI endpoints  
- KV cache reuse & batch scheduling  
- FSDP-style training step  
- CPU-only fallback logic  

---

# ⚡ 3. JAX / pmap mini example

```bash
python jax/mini_pmap_train.py --cpu-check
```

Verifies:

- JAX import  
- pmap usage  
- parallel mapping structure  

(If TPU available → remove the flag.)

---

# ⚡ 4. K8s & Terraform (infra skeleton)

```bash
kubectl apply --dry-run=client -f k8s/deployment.yaml
kubectl apply --dry-run=client -f k8s/service.yaml
```

Terraform:

```bash
cd terraform
terraform init
terraform validate
```

---

# ⚡ 5. FSDP-style tiny training loop

```bash
python training/fsdp_minimal.py
```

Works even without FSDP support (falls back automatically).

---

# 🎯 Design Principles

- **CPU-first reproducibility**  
- **Minimal, not trivial**  
- **Production-shaped APIs**  
- **Interview-ready structure**  
- **Tests = executable documentation**

---

# 📘 Example Notebook

```
examples/LLM-Infra-Lab.ipynb
```

---

# ⭐ If this repo helped you understand LLM infra, please consider starring it!
Stars help more engineers discover clean, readable infra examples.

---

# 🙌 Contributions welcome
