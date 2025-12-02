# execution_engine.py - Contains the high-level agent class, the stateless executor, and the main execution script

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import uuid
import json
import time
import traceback

# Import constants and core components
from config_consts import *
from core_architecture import GPlasticityMixin, EpisodicRelationalMemory, GPlasticDense, build_tcs26_model, finalize_tcs26

# Cell 9 — The Agent: TCS_GeneralIntelligence Class
class TCS_GeneralIntelligence(Model):
    """The Agent Wrapper that trains the core model."""
    def __init__(self, core_model, erm_buffer):
        super().__init__()
        self.core = core_model
        self.erm = erm_buffer
        self.sru = GPlasticDense(1, activation='sigmoid', name='sru_gate')
        self.ls_cls = tf.keras.losses.CategoricalCrossentropy()
        self.ls_mse = tf.keras.losses.MeanSquaredError()
        self.ls_cos = tf.keras.losses.CosineSimilarity()
        
        self.plastic_layers = [l for l in core_model.layers if isinstance(l, GPlasticityMixin)]

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.met_cls = tf.keras.metrics.Mean(name='loss_cls')
        self.met_sstc = tf.keras.metrics.Mean(name='loss_sstc')
        self.met_symp = tf.keras.metrics.Mean(name='loss_symplectic')

    @tf.function
    def train_step(self, data):
        x, y = data
        x_ins, x_ctrl = x[:21], x[21:]
        y_cls, y_st, y_rw, y_cnf, y_csl, y_hls, y_sstc, y_symp_target = y

        # Memory Retrieval
        q_vec = x_ins[18][0:1] # Use stream 18 as query
        mem_ctx = tf.tile(self.erm.retrieve_context(q_vec), [tf.shape(x_ins[0])[0], 1])
        
        with tf.GradientTape() as tape:
            # Forward
            preds = self.core(x_ins + list(x_ctrl) + [mem_ctx], training=True)
            p_cls, p_st, p_rw, _, p_cnf, p_rsa, p_csl, p_hls, p_mlc, p_ax, p_symp = preds
            
            # Losses
            l_cls = self.ls_cls(y_cls, p_cls)
            l_st = self.ls_mse(y_st, p_st)
            l_sstc = (1 + self.ls_cos(y_sstc, Concatenate()([p_csl, p_ax]))) * LOSS_WEIGHT_SSTC
            l_symp = self.ls_mse(y_symp_target, p_symp) * 0.5 # Physics loss
            
            total_loss = l_cls + l_st + l_sstc + l_symp + (tf.reduce_mean(p_mlc)*0.001)

        # Gradients & Plasticity
        grads = tape.gradient(total_loss, self.trainable_variables)
        
        # Apply G-Plasticity Modulations
        surp_sig = self.sru(Concatenate()([x_ctrl[0][0:1], l_st[None]]))
        for layer in self.plastic_layers:
            layer.surprisal_signal = surp_sig
            layer.dynamic_plasticity_strength = p_mlc
            delta, _ = layer.calculate_plasticity_change()
            # In a real loop, we'd adjust weights manually or via custom optimizer. 
            # For TF compat, we rely on standard backprop but modulate with signals if implemented deeply.

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Store Memory
        self.erm.store(Concatenate()([p_st, p_rw]), preds[3], p_csl)
        
        self.met_cls.update_state(l_cls)
        self.met_sstc.update_state(l_sstc)
        self.met_symp.update_state(l_symp)
        return {"cls": self.met_cls.result(), "sstc": self.met_sstc.result(), "symp": self.met_symp.result()}

# Cell 10 — Scalability Layer: Stateless Agent Executor
class StatelessAgentExecutor:
    """
    Simulates a horizontal scaling architecture (AgentOS/FastAPI pattern).
    Designed to load a TCS-26 instance, process a request statelessly, and detach.
    """
    def __init__(self, model_instance, memory_backend):
        self.model = model_instance
        self.memory = memory_backend
        self.id = str(uuid.uuid4())
        print(f"Agent Executor {self.id} online. Ready for stateless requests.")

    def process_request(self, input_payload):
        """
        Accepts a dictionary payload (simulating JSON), converts to tensors, 
        runs inference, and returns serializable JSON.
        """
        start_time = time.time()
        
        # 1. Parse Inputs (Simulated deserialization)
        # In a real FastAPI app, this handles pydantic models.
        # Here we assume input_payload contains prepared numpy arrays in a list.
        arrays = [np.array(a) for a in input_payload['data']]
        
        # 2. Memory Lookup (Simulating Redis call)
        # We use the internal memory object, but logically this is an external fetch.
        query = arrays[18][0:1] 
        mem_context = self.memory.retrieve_context(query)
        
        # 3. Inference
        # Reconstruct inputs: first 21 are data, next 3 are controls, last is memory
        full_inputs = arrays[:21] + arrays[21:] + [mem_context.numpy()]
        
        # Need to convert list of numpy arrays to list of tf.Tensors for predict
        inputs_list = [tf.constant(arr) for arr in full_inputs]

        results = self.model.predict(inputs_list, verbose=0)
        
        # 4. Response Formatting
        response = {
            "agent_id": self.id,
            "latency": time.time() - start_time,
            "classification": int(np.argmax(results[0][0])),
            "confidence": float(results[4][0][0]),
            "causal_state_mean": float(np.mean(results[6][0])),
            "symplectic_physics_vector": results[10][0].tolist()[:5] # Sample
        }
        return response

class MockFastAPI:
    """Mocking the API layer for the script demonstration."""
    def __init__(self, executor):
        self.executor = executor
    
    def post_predict(self, data):
        print(f"API: Received POST /predict from {data['client_ip']}")
        return self.executor.process_request(data['payload'])

# Cell 11 — Main Execution: Assembly, Training, and Scaled Inference
def gen_data(n=32):
    """Data generation helper function."""
    x = [np.random.rand(n, *shape).astype(np.float32) for shape in [
        IMAGE_SHAPE, (DATA_INPUT_SIZE,), (TS_STEPS, TS_DIM), (TS_STEPS, TS_DIM), 
        (TS_STEPS, TS_DIM), (TS_STEPS, TS_DIM), (SEQ_LEN,), (SEQ_LEN, SEQ_DIM),
        (32,32,1), (64,64,3), (16,16,8), (100,), (256,), (128,), (64,), (GRAPH_DIM**2,),
        (40,), (80,), (512,), (512,), (512,)]]
    
    # Text input (index 6) needs to be int
    x[6] = np.random.randint(0, VOCAB_SIZE, (n, SEQ_LEN))
    
    ctrl = [np.random.rand(n, 1).astype(np.float32), np.random.rand(n, 1).astype(np.float32), np.random.rand(n, 16).astype(np.float32)]
    
    y = [
        tf.one_hot(np.random.randint(0, NUM_CLASSES, n), NUM_CLASSES), # cls
        np.random.rand(n, FRONTAL_LOBE_UNITS*2 + CAUSAL_STATE_DIM).astype(np.float32), # state
        np.random.rand(n, 1).astype(np.float32), # rew
        np.random.rand(n, 1).astype(np.float32), # conf
        np.random.rand(n, CAUSAL_STATE_DIM).astype(np.float32), # causal
        np.random.rand(n, HYPER_LATENT_DIM).astype(np.float32), # hls
        np.random.rand(n, CAUSAL_STATE_DIM + AXIOMATIC_DIM).astype(np.float32), # sstc
        np.random.rand(n, SYMPLECTIC_DIM).astype(np.float32) # symplectic target
    ]
    return x + ctrl, y

def main_execution():
    print("🌌 --- INITIATING TCS-26 SYMPLECTIC-OMNI --- 🌌")

    # --- 2. Build ---
    try:
        inputs, controls, v, d, rsa = build_tcs26_model(IMAGE_SHAPE, DATA_INPUT_SIZE, TS_STEPS, TS_DIM, SEQ_LEN, SEQ_DIM, GRAPH_DIM, NUM_CLASSES)
        core = finalize_tcs26(inputs, controls, v, d, rsa)
        erm = EpisodicRelationalMemory()
        agent = TCS_GeneralIntelligence(core, erm)
        agent.compile(optimizer='adam')
        print("✅ Model Built. Symplectic Physics Engine: ACTIVE.")

        # --- 3. Train (Simulated) ---
        print("\n--- Training Loop (Epoch 1/1) ---")
        x_train, y_train = gen_data(TRAINING_BATCH_SIZE)
        
        # Keras .fit expects inputs as a list of Tensors/Arrays and targets as a list of Tensors/Arrays
        # The agent's train_step handles the memory retrieval internally.
        # We need to pass all X components (data+controls) and all Y components to .fit
        x_all_train = x_train[:21] + x_train[21:]
        
        # Match the y_train with the outputs expected by the model's train_step (8 targets)
        # Note: The original script's y_train only provided 8 components, which matches the required targets.
        y_all_train = y_train
        
        hist = agent.fit(x_all_train, y_all_train, epochs=1, verbose=1)
        
        # --- 4. Stateless Execution Demo ---
        print("\n--- Testing Horizontal Scalability (Mocked API) ---")
        executor = StatelessAgentExecutor(core, erm)
        api = MockFastAPI(executor)
        
        # Payload prep
        x_test, _ = gen_data(1)
        # The input data for the executor needs to be a flat list of all input arrays (data + controls)
        payload = {'client_ip': '192.168.1.X', 'payload': {'data': [arr.tolist() for arr in x_test]}}
        
        response = api.post_predict(payload)
        print("\n--- API RESPONSE ---")
        print(json.dumps(response, indent=2))
        print("\n✅ TCS-26 Execution Complete. Error Free. Now leave me alone.")

    except Exception as e:
        print(f"🚨 CRITICAL FAILURE: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main_execution()
