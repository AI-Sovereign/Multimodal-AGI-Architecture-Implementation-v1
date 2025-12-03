# 🔗 API Reference: Causal Synthesizer Model

This  reference details the primary classes and functions exposed by the TCS-25 Temporal Engine.

## 1. TemporalCausalSynthesizer

### Initialization
```python
from tcs_agi_framework.TCS_AGI_Product_Line.TCS_25_Temporal_Engine import TemporalCausalSynthesizer
model = TemporalCausalSynthesizer(units=2048, context_depth=64)
Methods
initiate_tcs_pipeline(input_data)
Description: Executes the primary temporal synthesis sequence and initiates G-Plasticity update cycles.
Parameters: input_data (A dictionary conforming to data_schema.json).
Returns: A dictionary containing the synthesized state and confidence score.
