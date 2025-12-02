# Imports all constants and makes them directly accessible
from .config_consts import (
    IMAGE_SHAPE, DATA_INPUT_SIZE, TS_STEPS, TS_DIM, SEQ_LEN, SEQ_DIM,
    GRAPH_DIM, VOCAB_SIZE, NUM_CLASSES, FRONTAL_LOBE_UNITS, HYPER_LATENT_DIM,
    NUM_PFC_CONTEXTS, RELATIONAL_EMB_DIM, CAUSAL_STATE_DIM, AXIOMATIC_DIM,
    MLC_OUTPUT_DIM, SYMPLECTIC_DIM, LOSS_WEIGHT_SSTC, TRAINING_BATCH_SIZE
)

# Imports all core architectural components (Layers, Mixins, Memory)
from .core_architecture import (
    GPlasticityMixin, GPlasticDense, GPlasticConv2D, GPlasticGRU,
    SymplecticIntegratorLayer, HTSP_Unit, AxiomaticKnowledgeLayer,
    EpisodicRelationalMemory, RelationalSelfAttention, CGWAttentionLayer,
    MetaLearningControl, BasalGangliaSelectionLayer, MultiContextExecutiveGating,
    CausalInferenceModule, HLS_Processor, HLS_Decoder,
    extract_and_hybrid, build_tcs26_model, finalize_tcs26
)

# Imports the main Agent and Execution Wrappers
from .execution_engine import (
    TCS_GeneralIntelligence, StatelessAgentExecutor, MockFastAPI
)

# Define the package's public API explicitly (optional but good practice)
__all__ = [
    # Constants
    "IMAGE_SHAPE", "DATA_INPUT_SIZE", "SYMPLECTIC_DIM", "FRONTAL_LOBE_UNITS",
    "NUM_CLASSES", "CAUSAL_STATE_DIM", "AXIOMATIC_DIM", "HYPER_LATENT_DIM",
    
    # Layers & Mixins
    "GPlasticityMixin", "GPlasticDense", "SymplecticIntegratorLayer",
    "RelationalSelfAttention", "CGWAttentionLayer", "CausalInferenceModule",
    "EpisodicRelationalMemory", "TCS_GeneralIntelligence",
    
    # Core Functions
    "build_tcs26_model", "finalize_tcs26",
    
    # Execution
    "StatelessAgentExecutor", "MockFastAPI"
]

print("TCS-26 Symplectic-Omni Package: Core modules loaded and linked.")
