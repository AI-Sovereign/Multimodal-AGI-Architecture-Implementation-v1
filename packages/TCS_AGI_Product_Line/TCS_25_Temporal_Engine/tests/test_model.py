import unittest
import tensorflow as tf
from config_consts import *
from core_architecture import build_tcs_net_core_model, EpisodicRelationalMemory

class TestTCSModel(unittest.TestCase):
    def test_build_model(self):
        """Test if the core model builds without error and has correct output count."""
        try:
            model = build_tcs_net_core_model(IMAGE_SHAPE, DATA_INPUT_SIZE, TS_STEPS, TS_DIM, SEQ_LEN, SEQ_DIM, GRAPH_DIM, NUM_CLASSES)
            self.assertIsInstance(model, tf.keras.models.Model)
            # Check number of outputs (should be 10 based on the code)
            self.assertEqual(len(model.outputs), 10)
        except Exception as e:
            self.fail(f"Model build failed: {e}")

    def test_memory_module(self):
        """Test basic Episodic Relational Memory instantiation."""
        memory = EpisodicRelationalMemory()
        self.assertIsInstance(memory, EpisodicRelationalMemory)
        self.assertEqual(len(memory.buffer), 0)

if __name__ == '__main__':
    unittest.main()
