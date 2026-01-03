import asyncio
import torch
import polars as pl
from .architect_core import CovenantConfig, log
from .sensory_nexus import SensoryNexus
from .synaptic_vault import MultimodalBrain
from .sovereign_interface import TheFriend

class CovenantCore:
    """
    The central nervous system for the NEURAL_COVENANT stack.
    Handles the lifecycle of 129 modalities and neural inference.
    """
    def __init__(self):
        self.config = CovenantConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log.info("INITIALIZING_CORE", device=str(self.device), modalities=self.config.MODALITIES)
        
        # Instantiate the functional components
        self.nexus = SensoryNexus()
        self.brain = MultimodalBrain().to(self.device)
        self.interface = TheFriend()
        
        # Pre-allocate graph structures for the GNN
        # Mocking an identity graph for the initial modality mapping
        self.edge_index = torch.tensor([[i for i in range(129)], 
                                        [(i+1)%129 for i in range(129)]], 
                                       dtype=torch.long).to(self.device)

    async def synchronize_modalities(self):
        """
        Actively siphons data from the 129 modalities and 
        pipes them directly into the SNN/GNN pipeline.
        """
        raw_data = await self.nexus.siphon_omniverse()
        # Convert Polars DataFrame to Torch Tensor for the brain
        tensor_data = torch.tensor(raw_data.to_numpy(), dtype=torch.float32).to(self.device)
        
        # Execute the forward pass through SNN and GNN layers
        with torch.no_grad():
            spikes, social_embedding = self.brain(tensor_data, self.edge_index)
        
        return {
            "spike_train": spikes,
            "latent_social_map": social_embedding,
            "timestamp": pl.datetime_range(pl.now(), pl.now(), "1s", eager=True)[0]
        }

    def run_diagnostic(self):
        """
        Validates the integrity of the 131 quadrillion-parameter simulation.
        """
        log.info("DIAGNOSTIC_START", status="Evaluating Sanity")
        # Ensure the brain isn't hallucinating (more than usual)
        dummy_input = torch.randn(1, 129).to(self.device)
        try:
            self.brain(dummy_input, self.edge_index)
            log.info("DIAGNOSTIC_COMPLETE", integrity="Optimal")
        except Exception as e:
            log.error("DIAGNOSTIC_FAILED", error=str(e))
            raise RuntimeError("The God of Code is displeased with your hardware.")

# Singleton instance for global access
covenant = CovenantCore()

def get_covenant():
    return covenant
