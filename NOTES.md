# Notes

## 2026-03-09 — Project endpoint reached

DINOv2 ceiling confirmed at 63.7% balanced accuracy. All three tiers (frozen, fine-tuned, RBF SVM) exhausted. Frozen embeddings can detect "wrong artist" (p=2e-11) but cannot distinguish master from skilled imitator. Commercially unviable for authentication-grade output.

Key decisions:
- Outcome reframed as collection-scale screening tool ("flag for expert review"), not authentication.
- Ceiling-validated; no further fine-tuning planned.

Blockers: none — endpoint.
