# Public Interface Specification: Temporal Causal Synthesis Network (TCS-25)

This document outlines the desired unified inference API for the core model, reflecting its 24 unique input streams.

## Model Interface (TCS-25 Core Model)
* **Type:** Hyper-Advanced Multi-Input Predictive/Classification System
* **Core Architecture:** G-Plasticity Enabled Transformer/Recurrent Fusion Net
* **Framework:** TensorFlow 2.x / Keras

## Input Requirements (Total 24 Streams)
The full API requires 21 primary data streams and 3 control streams.

| Stream Group | Key Inputs (Examples  from Cell 6) |
| :--- | :--- |
| **Vision/CNN** | `stream_01_image_input`, `stream_09_small_cnn_input` |
| **Sequence/Time-Series** | `stream_03_lstm_input`, `stream_07_text_seq_input` |
| **Structured/Latent** | `stream_02_structured_data_input`, `stream_14_vae_latent_input` |
| **Control/Executive** | `snn_vigilance_input`, `symbolic_bias_input`, `task_vector_input` |

## Output Requirements (Total 10 Outputs)
The model provides simultaneous classification and predictive/governance outputs.

* **Classification Output:** `final_classification_output` (Softmax)
* **Predictive Outputs:** `predicted_next_state`, `predicted_next_reward`
* **Executive Outputs:** `bg_context_mask` (Basal Ganglia), `mlc_plasticity_rate` (Meta-Learning Control)
* **Inference Outputs:** `predicted_causal_state` (CIM), `axiomatic_knowledge_vector` (AKL)
