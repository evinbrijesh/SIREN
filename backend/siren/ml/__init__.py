"""SIREN ML module — Siamese U-Net change detection + SegFormer classifier.

Optional evidence layer (ADR-002). Falls back to the deterministic rule-based
mask when torch is unavailable or no trained weights are found. Never replaces
the deterministic pipeline in the critical path (Hard Rule 1).
"""
