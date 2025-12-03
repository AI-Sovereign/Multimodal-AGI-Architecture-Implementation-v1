# Model Card: TCS 25 (Temporal Causal Synthesis Network)

## Model Details
* **Developer:** Advanced AI Professional
* **Date:** November 2025
* **Version:** 0.9.1-beta
* **Framework:** TensorFlow/Keras
* **Type:** **G-Plastic Multimodal Causal Network (RNN/Attention Fusion with Episodic Memory)**

## Key Architectural Features (Extracted from Code)
* **General Plasticity (G-Plasticity):** Weights modulated by `surprisal_signal` and `causal_signal`.
* **Conscious Global Workspace (CGW):** Fuses 6 high-level vectors including HLS and Axiomatic Knowledge.
* **Causal Inference Module (CIM):** Predicts causal state (`CAUSAL_STATE_DIM=256`).
* **Relational Self-Attention (RSA):** Dynamically scales 21 feature streams into Ventral/Dorsal paths.
* **Hyper-Latent Space (HLS):** Sparse, massive latent space (4096 dimensions).

## Intended Use
* **Primary Use:** General Intelligence Research, Multimodal Time-Series Prediction, and Dynamic Contextual Classification.
* **Out of Scope:** Use in high-stakes environments (e.g., medical diagnosis, financial trading).

## Metrics (Loss Weights and Training Targets)
| Metric | Focus | Weight/Dimension |
| :--- | :--- | :--- |
| **LOSS_WEIGHT_SSTC** | Self-Supervised Temporal Contrastive Loss | 0.95 (High Priority) |
| State Prediction Loss | `tf.keras.losses.MeanSquaredError` | PWM_STATE_DIM = 5344 |
| Classification Loss | `tf.keras.losses.CategoricalCrossentropy` | NUM_CLASSES = 10 |
