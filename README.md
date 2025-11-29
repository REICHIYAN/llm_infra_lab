# LLM-Infra-Lab

A minimal yet realistic LLM infrastructure lab designed under the constraints:

- **Free to run**
- **Verifiable on a single CPU-only laptop**
- **Implementable in ~2 hours by a single engineer**
- **Covers 4 key areas that Google SWE / Research Engineers care about**:

1. JAX / TPU / distributed training patterns (`pmap`)
2. vLLM-style fast inference: KV cache & batching design
3. Kubernetes / GKE + Terraform deployment skeleton
4. FSDP-based large-model training pattern (CPU-friendly demo)

The goal is not peak performance, but **clean architecture, correctness, and easy verification**.
Everything runs on CPU; TPU/GPU usage is optional and not required for basic checks.

---

## Project Layout

```text
llm_infra_lab/
├─ README.md
├─ requirements.txt
├─ jax/
│   └─ mini_pmap_train.py      # JAX + pmap mini example (CPU check + TPU-ready)
├─ serving/
│   ├─ __init__.py
│   ├─ api_server.py           # FastAPI app: /health, /generate
│   └─ vllm_mock.py            # vLLM-style KV cache & batching mock (CPU-only)
├─ training/
│   ├─ __init__.py
│   └─ fsdp_minimal.py         # FSDP-style tiny training loop (CPU single-process)
├─ tests/
│   ├─ test_serving.py         # API tests (FastAPI TestClient)
│   ├─ test_vllm_mock.py       # KV cache & batching behavior tests
│   └─ test_fsdp.py            # Single-step FSDP training test
├─ k8s/
│   ├─ deployment.yaml         # K8s Deployment (GKE-ready, but generic)
│   └─ service.yaml            # ClusterIP Service for the API
├─ terraform/
│   └─ main.tf                 # Minimal GKE cluster definition (validate-only)
└─ scripts/
    ├─ run_server.sh           # Start FastAPI (development use)
    └─ run_tests.sh            # Run pytest test suite
```

---

## 0. Setup (CPU-only)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

All dependencies are CPU-friendly: no CUDA or TPU is required.

---

## 1. Run the API server (serving layer)

```bash
./scripts/run_server.sh
```

Then in another terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate       -H "Content-Type: application/json"       -d '{"prompt": "Hello from LLM-Infra-Lab!", "max_new_tokens": 32}'
```

This exercises:

- `serving/api_server.py` (FastAPI)
- `serving/vllm_mock.py` (KV cache & batching mock)

---

## 2. Run tests (quick verification)

```bash
./scripts/run_tests.sh
# or
pytest -q
```

This validates:

- API endpoints (`/health`, `/generate`)
- KV cache reuse and batching behavior
- FSDP-style training step on CPU (fallback to non-FSDP if unavailable)

---

## 3. JAX / pmap mini example (CPU check)

The JAX example is in `jax/mini_pmap_train.py`.

On a CPU-only machine, you can run a **lightweight structural check**:

```bash
python jax/mini_pmap_train.py --cpu-check
```

This confirms:

- JAX is installed and importable
- `pmap` is used correctly
- The function signatures & parallel mapping structure are valid

On Colab with TPU, you can omit `--cpu-check` and execute the full mini-training loop.

---

## 4. Kubernetes / Terraform (validate-only)

These files are designed to be:

- **Syntactically valid**
- **Logically minimal**
- **Safe to inspect and validate without creating real cloud resources**

If you have `kubectl` installed:

```bash
kubectl apply --dry-run=client -f k8s/deployment.yaml
kubectl apply --dry-run=client -f k8s/service.yaml
```

If you have `terraform` installed:

```bash
cd terraform
terraform init
terraform validate
```

This is sufficient for interview or review contexts to show that:

- You understand basic K8s Deployment/Service structure
- You can express infrastructure as code for a GKE cluster

---

## 5. FSDP-style minimal training

```bash
python training/fsdp_minimal.py
```

This script attempts to initialize a single-process CPU-only process group and wrap
a tiny model in FSDP. If FSDP or distributed backends are not available in your
PyTorch build, it will **gracefully fall back** to a non-FSDP model while keeping
the code structure intact.

The goal is to demonstrate that you:

- Know how FSDP is wired conceptually
- Can write a training step that works in both FSDP and non-FSDP modes

---

## Design Philosophy (for interviews)

- **CPU-first**: Everything important can be verified on a CPU-only machine.
- **Clear separation of concerns**:
  - `serving/`: inference API & fast-path considerations
  - `training/`: large-model training patterns (FSDP-style)
  - `jax/`: JAX-based research-style experimentation
  - `k8s/`, `terraform/`: deployment & infra-as-code
  - `tests/`: executable documentation & safety net
- **Minimal, not trivial**: Each file is intentionally small but structurally realistic,
  so a senior engineer can quickly assess correctness, and you can speak about it
  clearly in an interview.