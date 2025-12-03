# test_model.py - Contains unit tests for the TCS-26 architecture and execution engine

import unittest
import tensorflow as tf
import numpy as np
import random # Needed for gen_data seed

# Import constants and core components
from config_consts import *
from core_architecture import build_tcs26_model, finalize_tcs26, EpisodicRelationalMemory, GPlasticityMixin
from execution_engine import TCS_GeneralIntelligence, gen_data # Import gen_data from execution_engine

# Helper to create inputs for the full model
def create_mock_inputs():
    """Generates a single batch of mock data for testing model inference."""
    # Use gen_data which handles all 24 inputs (21 data + 3 controls)
    x_data, _ = gen_data(n=1) 
    
    # Separate data inputs (21) and controls (3)
    x_ins = x_data[:21]
    x_ctrl = x_data[21:]
    
    # Generate mock memory context
    retrieved_state = np.random.rand(1, FRONTAL_LOBE_UNITS).astype(np.float32)
    retrieved_causal = np.random.rand(1, CAUSAL_STATE_DIM).astype(np.float32)
    mem_context = np.concatenate([retrieved_state, retrieved_causal], axis=-1)
    
    # Full inputs are data (21) + controls (3) + memory context (1) = 25 inputs
    full_inputs = x_ins + x_ctrl + [mem_context]
    return full_inputs

class TestTCS26Model(unittest.TestCase):
    
    # Store core model for reuse across tests
    @classmethod
    def setUpClass(cls):
        """Build the model once for all tests."""
        # Force NumPy seed for deterministic tests
        np.random.seed(42) 
        random.seed(42) 
        
        inputs, controls, v, d, rsa = build_tcs26_model(
            IMAGE_SHAPE, DATA_INPUT_SIZE, TS_STEPS, TS_DIM, SEQ_LEN, SEQ_DIM, 
            GRAPH_DIM, NUM_CLASSES
        )
        cls.core_model = finalize_tcs26(inputs, controls, v, d, rsa)
        cls.erm = EpisodicRelationalMemory()
        cls.agent = TCS_GeneralIntelligence(cls.core_model, cls.erm)
        cls.agent.compile(optimizer=tf.keras.optimizers.Adam())

    def test_01_build_and_output_count(self):
        """Test if the core model builds and has the correct number of outputs."""
        self.assertIsInstance(self.core_model, tf.keras.models.Model, "Model is not a Keras Model instance.")
        # TCS-26 outputs 11 tensors: [cls, state, rew, bg_mask, mcc_conf, rsa_weights, causal, hls_raw, mlc_out, axiom, symplectic_evolved]
        self.assertEqual(len(self.core_model.outputs), 11, f"Expected 11 outputs, but got {len(self.core_model.outputs)}.")
        self.assertEqual(self.core_model.name, 'TCS26_Symplectic_Omni', "Model name mismatch.")
        
    def test_02_memory_module_integrity(self):
        """Test basic Episodic Relational Memory instantiation and functionality."""
        self.assertIsInstance(self.erm, EpisodicRelationalMemory, "Memory is not an EpisodicRelationalMemory instance.")
        self.assertEqual(len(self.erm.buffer), 0, "Memory buffer should start empty.")
        
    def test_03_inference_run(self):
        """Test if the model runs inference without crashing and outputs correct shapes."""
        try:
            full_inputs = create_mock_inputs()
            
            # Predict requires a list of arrays
            inputs_list = [tf.constant(arr) for arr in full_inputs]
            
            preds = self.core_model.predict(inputs_list, verbose=0)
            
            # Check a few critical output shapes (batch size 1)
            self.assertEqual(preds[0].shape, (1, NUM_CLASSES), "Classification output shape incorrect.") # out_cls
            self.assertEqual(preds[6].shape, (1, CAUSAL_STATE_DIM), "Causal state output shape incorrect.") # out_causal
            self.assertEqual(preds[10].shape, (1, SYMPLECTIC_DIM), "Symplectic output shape incorrect.") # symplectic_evolved
        except Exception as e:
            self.fail(f"Model inference failed during run: {e}")

    def test_04_plasticity_layer_count(self):
        """Test if GPlasticityMixin layers were correctly implemented and counted."""
        plastic_layers_count = len(self.agent.plastic_layers)
        # Check against expected count (which is high due to ubiquitous use)
        self.assertGreater(plastic_layers_count, 90, f"Expected >90 GPlasticityMixin layers, found {plastic_layers_count}.")
        
    def test_05_train_step_functionality(self):
        """Test if a single train step executes without errors and updates metrics."""
        # Use a consistent seed for data generation
        np.random.seed(42) 
        random.seed(42)
        x_train, y_train = gen_data(n=TRAINING_BATCH_SIZE)
        
        # Prepare data for train_step
        x_all = x_train[:21] + x_train[21:]
        data_tuple = (x_all, y_train)
        
        # Manually run one train step (requires memory context input to the core model, which train_step handles)
        results = self.agent.train_step(data_tuple) 
        
        self.assertIn("cls", results, "Classification loss metric missing from results.")
        self.assertIn("symp", results, "Symplectic loss metric missing from results.")
        # Check if loss is a tensor/numpy float
        self.assertTrue(tf.is_tensor(results['cls']) or isinstance(results['cls'], np.float32), "Loss result is not a float/tensor.")
        
        # Reset seed for next test run if needed, though setUpClass handles this for now
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
