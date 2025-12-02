# core_architecture.py - Contains all custom layers and the model assembly functions

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, Flatten, Embedding, LSTM, GRU, Bidirectional, TimeDistributed, GlobalMaxPooling1D, Concatenate, Multiply, BatchNormalization, Dot, Reshape, Dropout, Attention, Add, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import random

# Import constants from the config file
from config_consts import *

# Cell 2 — G-Plasticity Engine (The "Brain" Glue)
class GPlasticityMixin:
    """TCS-26 Base Mixin: Generalized Plasticity with Dynamic Modulation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surprisal_signal = tf.constant([[0.0]], dtype=tf.float32)
        self.causal_signal = tf.constant([[0.0]], dtype=tf.float32)    
        self.dynamic_plasticity_strength = tf.constant([[0.0001]], dtype=tf.float32)
        self.w_old = None
        self.x_input = None
        self.y_output = None

    def calculate_plasticity_change(self):
        if self.w_old is None or self.x_input is None or self.y_output is None:
            return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

        # Hebbian calculation: Outer product of input and output
        hebbian_term = tf.matmul(tf.transpose(self.x_input), self.y_output) / tf.cast(tf.shape(self.x_input)[0], tf.float32)
        
        # Modulated by global surprise and causal error
        modulation_scalar = 0.6 * tf.cast(self.surprisal_signal[0, 0], tf.float32) + 0.4 * tf.cast(self.causal_signal[0, 0], tf.float32)
        delta_p = tf.cast(self.dynamic_plasticity_strength[0, 0], tf.float32) * modulation_scalar * hebbian_term
        return delta_p, tf.norm(delta_p)

    def track_activations(self, x_input, y_output):
        self.x_input = tf.stop_gradient(x_input)
        self.y_output = tf.stop_gradient(y_output)

class GPlasticDense(GPlasticityMixin, Dense):
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(units=units, activation=activation, **kwargs)
    def call(self, inputs):
        self.w_old = self.weights[0] if self.weights else None
        output = super().call(inputs)
        self.track_activations(inputs, output)
        return output

class GPlasticConv2D(GPlasticityMixin, Conv2D):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, activation=activation, **kwargs)
    def call(self, inputs):
        self.w_old = self.weights[0] if self.weights else None
        output = super().call(inputs)
        flat_out = Flatten()(output)
        self.track_activations(inputs, flat_out)
        return output

class GPlasticGRU(GPlasticityMixin, Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.gru_cell = GRU(units, return_sequences=False, return_state=True)
        self.proj_input = GPlasticDense(units, activation='tanh', name='gplastic_gru_input_proj')
    def call(self, inputs, initial_state=None):
        projected_input = self.proj_input(inputs) 
        output, state = self.gru_cell(projected_input, initial_state=initial_state)
        self.w_old = self.gru_cell.weights[0] if self.gru_cell.weights else None
        return output, state

# Cell 3 — Symplectic Neural Network Layer (New Physics Engine)
class SymplecticIntegratorLayer(Layer):
    """
    Implements a Symplectic Map to preserve phase space volume in latent dynamics.
    Splits input into (q, p) coordinates and applies a Hamiltonian-inspired update.
    Advanced upgrade from TCS-25.
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        # Hamiltonian derivative approximators
        self.mlp_q = GPlasticDense(self.half_dim, activation='tanh', name='symplectic_H_dq')
        self.mlp_p = GPlasticDense(self.half_dim, activation='tanh', name='symplectic_H_dp')
        self.dt = 0.1 # Time step

    def call(self, inputs):
        # Expect inputs of shape (batch, dim)
        q = inputs[:, :self.half_dim]
        p = inputs[:, self.half_dim:]

        # Symplectic Euler update (Volume preserving)
        # p_{t+1} = p_t - dt * dH/dq
        dp = self.mlp_q(q) 
        p_new = p - self.dt * dp

        # q_{t+1} = q_t + dt * dH/dp_{t+1}
        dq = self.mlp_p(p_new)
        q_new = q + self.dt * dq

        return Concatenate()([q_new, p_new])

class HTSP_Unit(Layer):
    """Hierarchical Temporal Scale Processor (from TCS-25)"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.fast_gru = GRU(units // 2, return_sequences=False, return_state=True)
        self.slow_lstm = LSTM(units // 2, return_sequences=False, return_state=True)
        self.dense_gate = GPlasticDense(units, activation='sigmoid', name='htsp_gate_plastic')
    def call(self, inputs, fast_state, slow_state):
        fast_output, fast_state = self.fast_gru(inputs, initial_state=fast_state)
        slow_output, slow_state_h, slow_state_c = self.slow_lstm(inputs, initial_state=slow_state)
        slow_state = [slow_state_h, slow_state_c]
        fused = Concatenate()([fast_output, slow_output])
        gate = self.dense_gate(fused)
        output = Multiply()([fused, gate])
        return output, fast_state, slow_state

# Cell 4 — Advanced Memory (Axiomatic & Episodic)
class AxiomaticKnowledgeLayer(Layer):
    """Injects structured truths."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.axiom_embeddings = self.add_weight(
            name="axiom_emb_matrix",
            shape=(NUM_PFC_CONTEXTS, AXIOMATIC_DIM),
            initializer='glorot_uniform', trainable=True
        )
    def call(self, context_mask):
        expanded_mask = tf.expand_dims(context_mask, axis=-1)
        weighted_axioms = expanded_mask * self.axiom_embeddings
        return tf.reduce_sum(weighted_axioms, axis=1)

class EpisodicRelationalMemory:
    """Graph-based memory designed for later serialization (Stateless support)."""
    def __init__(self):
        self.buffer = []
        self.max_size = 10000
        self.graph_nodes = {}
        self.graph_edges = {}

    def store(self, state_vector, context_mask, causal_vector):
        # Convert tensors to numpy for storage
        if tf.is_tensor(state_vector): state_vector = state_vector.numpy()
        if tf.is_tensor(context_mask): context_mask = context_mask.numpy()
        if tf.is_tensor(causal_vector): causal_vector = causal_vector.numpy()
        
        node_id = len(self.buffer)
        episode = {'state': state_vector[0], 'context': context_mask[0], 'causal': causal_vector[0], 'id': node_id}
        self.buffer.append(episode)
        self.graph_nodes[node_id] = episode

        if node_id > 0:
            past_node_id = random.randint(0, node_id - 1)
            self.graph_edges.setdefault(past_node_id, []).append(node_id)

        if len(self.buffer) > self.max_size:
            old_node = self.buffer.pop(0)
            del self.graph_nodes[old_node['id']]
            if old_node['id'] in self.graph_edges: del self.graph_edges[old_node['id']]

    def retrieve_context(self, query_vector):
        if not self.graph_nodes:
            return tf.zeros((1, FRONTAL_LOBE_UNITS + CAUSAL_STATE_DIM), dtype=tf.float32)
        query = np.asarray(query_vector[0])
        # Simple similarity retrieval
        scores, node_list = [], list(self.graph_nodes.values())
        for node in node_list:
            score = np.dot(query, node['state'][:len(query)])
            scores.append(score)
        
        # Softmax weighting
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / np.sum(exp_scores)
        
        retrieved_state = np.zeros(FRONTAL_LOBE_UNITS, dtype=np.float32)
        retrieved_causal = np.zeros(CAUSAL_STATE_DIM, dtype=np.float32)
        
        for i, node in enumerate(node_list):
            retrieved_state += weights[i] * node['state'][:FRONTAL_LOBE_UNITS]
            retrieved_causal += weights[i] * node['causal']
            
        return Concatenate()([
            tf.constant(retrieved_state[None, :], dtype=tf.float32), 
            tf.constant(retrieved_causal[None, :], dtype=tf.float32)
        ])

# Cell 5 — Relational Attention (RSA)
class RelationalSelfAttention(Layer):
    """
    Dynamically scales features based on global context.
    Essential for determining which input stream matters.
    """
    def __init__(self, num_features, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.query_gen = GPlasticDense(feature_dim, activation='relu', name='rsa_query_plastic')
        self.key_gen = GPlasticDense(feature_dim, activation='relu', name='rsa_key_plastic')
        self.rel_context_proj_gate = GPlasticDense(self.num_features * self.num_features, activation='tanh', name='rel_context_proj_gate')
        self.attention_weights_generator = GPlasticDense(num_features, activation='softmax', name='rsa_weights_plastic')

    def call(self, projected_feature_list):
        global_context_fused = Concatenate()(projected_feature_list)
        queries = tf.stack([self.query_gen(f) for f in projected_feature_list], axis=1)
        keys = tf.stack([self.key_gen(f) for f in projected_feature_list], axis=1)
        
        # Standard attention scores
        attention_scores = Dot(axes=-1)([queries, keys])
        
        # Global context gating
        relational_context_proj = self.rel_context_proj_gate(global_context_fused)
        combined_context = Concatenate()([global_context_fused, relational_context_proj])
        
        attention_weights = self.attention_weights_generator(combined_context)
        
        attended_features = []
        for i, feature in enumerate(projected_feature_list):
            weight = attention_weights[:, i:i+1]
            attended_features.append(Multiply()([feature, weight]))
            
        return attended_features, attention_weights, tf.reduce_mean(attention_scores)

# Cell 6 — Executive Control Systems (CGW, MLC, BG, CIM)
class CGWAttentionLayer(Layer):
    """Conscious Global Workspace: The bottleneck of awareness."""
    def build(self, input_shape):
        fusion_dim = sum([shape[-1] for shape in input_shape])
        self.kernel = self.add_weight(name="kernel_fusion", shape=(fusion_dim, input_shape[0][-1]), initializer='glorot_uniform', trainable=True)
        self.gating_dense = GPlasticDense(1, activation='sigmoid', name='cgw_gating_plastic')
        super().build(input_shape)

    def call(self, inputs):
        # inputs: [plastic_gru, symbolic, vigilance, mcc, hls, axiom, symplectic]
        fused = Concatenate()(inputs)
        gated_fused = Multiply()([fused, self.gating_dense(fused)])
        return tf.matmul(gated_fused, self.kernel)

class MetaLearningControl(Layer):
    """Regulates the plasticity rate dynamically."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_pred = GPlasticDense(MLC_OUTPUT_DIM, activation='sigmoid', name='mlc_output_gplastic')
        self.dropout = Dropout(0.2)
    def call(self, pfc_gated_output):
        return self.dense_pred(self.dropout(pfc_gated_output))

class BasalGangliaSelectionLayer(Layer):
    """Selects the active context/task rule."""
    def __init__(self, num_contexts, **kwargs):
        super().__init__(**kwargs)
        self.context_gate = GPlasticDense(num_contexts, activation='softmax', name='bg_context_gate_gplastic')
    def call(self, cgw_output, task_vector_input):
        return self.context_gate(Concatenate()([cgw_output, task_vector_input]))

class MultiContextExecutiveGating(Layer):
    """PFC: Routes signals based on BG selection."""
    def __init__(self, units, num_contexts, **kwargs):
        super().__init__(**kwargs)
        self.context_networks = [GPlasticDense(units, activation='tanh', name=f'pfc_context_gplastic_{i}') for i in range(num_contexts)]
    def call(self, cgw_output, context_mask):
        context_outputs = tf.stack([net(cgw_output) for net in self.context_networks], axis=1)
        weighted_mask = tf.expand_dims(context_mask, axis=-1)
        return tf.reduce_sum(Multiply()([context_outputs, weighted_mask]), axis=1)

class CausalInferenceModule(Layer):
    """Deduced causality from state transitions."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = GPlasticDense(CAUSAL_STATE_DIM, activation='sigmoid', name='cim_causal_output_gplastic')
        self.gating = GPlasticDense(CAUSAL_STATE_DIM, activation='tanh', name='cim_gating_gplastic')
    def call(self, pfc_gated_output, retrieved_causal_context):
        fused = Concatenate()([pfc_gated_output, retrieved_causal_context])
        return Multiply()([self.dense(fused), self.gating(fused)])

# Cell 7 — HLS Processor & Feature Extraction
class HLS_Processor(Layer):
    """Hyper-Latent Space Processor."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_proj = GPlasticDense(FRONTAL_LOBE_UNITS * 2, activation='relu', name='hls_input_proj_plastic')
        self.latent_dense = GPlasticDense(HYPER_LATENT_DIM, activation='relu', name='hls_core_plastic', use_bias=False)
    def call(self, pfc_output):
        return self.latent_dense(self.input_proj(pfc_output))

class HLS_Decoder(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = GPlasticDense(FRONTAL_LOBE_UNITS // 2, activation='tanh', name='hls_decoder_plastic')
    def call(self, hls_vector):
        return self.decoder(hls_vector)

def extract_and_hybrid(input_tensor, units, name):
    """Helper to attach GPlastic layer to feature streams."""
    return GPlasticDense(units, activation='tanh', name=f'hybrid_synapse_gplastic_{name}')(input_tensor)

# --- TCS-26 Core Model Construction Start (Cell 7) ---
def build_tcs26_model(image_shape, data_size, ts_steps, ts_dim, seq_len, seq_dim, graph_dim, num_classes):
    # --- Input Definition (23 Streams + Controls) ---
    s1_img = Input(shape=image_shape, name='s01_img')
    s2_data = Input(shape=(data_size,), name='s02_data')
    s3_lstm = Input(shape=(ts_steps, ts_dim), name='s03_lstm')
    s4_gru = Input(shape=(ts_steps, ts_dim), name='s04_gru')
    s5_bi = Input(shape=(ts_steps, ts_dim), name='s05_bi')
    s6_td = Input(shape=(ts_steps, ts_dim), name='s06_td')
    s7_text = Input(shape=(seq_len,), name='s07_text')
    s8_trans = Input(shape=(seq_len, seq_dim), name='s08_trans')
    s9_cnn_s = Input(shape=(32, 32, 1), name='s09_cnn_s')
    s10_cnn_d = Input(shape=(64, 64, 3), name='s10_cnn_d')
    s11_ae = Input(shape=(16, 16, 8), name='s11_ae')
    s12_fnn1 = Input(shape=(100,), name='s12_fnn1')
    s13_fnn2 = Input(shape=(256,), name='s13_fnn2')
    s14_vae = Input(shape=(128,), name='s14_vae')
    s15_rbm = Input(shape=(64,), name='s15_rbm')
    s16_graph = Input(shape=(graph_dim * graph_dim,), name='s16_graph')
    s17_attn = Input(shape=(40,), name='s17_attn')
    s18_cust = Input(shape=(80,), name='s18_cust')
    s19_res_f = Input(shape=(512,), name='s19_res_f')
    s20_res_b = Input(shape=(512,), name='s20_res_b')
    s21_res_final = Input(shape=(512,), name='s21_res_final')
    
    # Controls
    snn_vigilance = Input(shape=(1,), name='ctrl_vigilance')
    symbolic_bias = Input(shape=(1,), name='ctrl_bias')
    task_vector = Input(shape=(16,), name='ctrl_task')
    retrieved_mem = Input(shape=(FRONTAL_LOBE_UNITS + CAUSAL_STATE_DIM,), name='ctrl_memory')

    # --- Feature Extraction ---
    f1 = extract_and_hybrid(Flatten()(GPlasticConv2D(128, 3, activation='relu')(s1_img)), 512, 'img')
    f2 = extract_and_hybrid(s2_data, 256, 'struct')
    f3 = extract_and_hybrid(LSTM(256)(s3_lstm), 512, 'lstm')
    f4 = extract_and_hybrid(GRU(256)(s4_gru), 512, 'gru')
    f5 = extract_and_hybrid(Bidirectional(LSTM(256))(s5_bi), 512, 'bi')
    f6 = extract_and_hybrid(GlobalMaxPooling1D()(TimeDistributed(GPlasticDense(128))(s6_td)), 256, 'td')
    f7 = extract_and_hybrid(GlobalMaxPooling1D()(Embedding(VOCAB_SIZE, 128)(s7_text)), 256, 'txt')
    f8 = extract_and_hybrid(Flatten()(s8_trans), 512, 'trans')
    f9 = extract_and_hybrid(Flatten()(GPlasticConv2D(64,3)(s9_cnn_s)), 256, 'cnns')
    f10 = extract_and_hybrid(Flatten()(GPlasticConv2D(128,3)(s10_cnn_d)), 512, 'cnnd')
    f11 = extract_and_hybrid(Flatten()(GPlasticConv2D(64,3)(s11_ae)), 256, 'ae')
    f12 = extract_and_hybrid(s12_fnn1, 256, 'fnn1')
    f13 = extract_and_hybrid(s13_fnn2, 512, 'fnn2')
    f14 = extract_and_hybrid(s14_vae, 512, 'vae')
    f15 = extract_and_hybrid(s15_rbm, 256, 'rbm')
    f16 = extract_and_hybrid(s16_graph, 512, 'graph')
    f17 = extract_and_hybrid(s17_attn, 256, 'attn')
    f18 = extract_and_hybrid(s18_cust, 512, 'cust')
    f19 = extract_and_hybrid(s19_res_f, 512, 'resf')
    f20 = extract_and_hybrid(s20_res_b, 512, 'resb')
    f21 = extract_and_hybrid(s21_res_final, 512, 'resfin')

    all_raw = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21]
    
    # --- RSA ---
    proj_feats = [GPlasticDense(512, activation='relu')(f) for f in all_raw]
    rsa_feats, rsa_weights, _ = RelationalSelfAttention(21, 512)(proj_feats)
    
    # Split Ventral/Dorsal
    ventral = Concatenate()([rsa_feats[i] for i in [0,6,7,8,9,10,1,11,12,13,14,16]])
    dorsal = Concatenate()([rsa_feats[i] for i in [2,3,4,5,15,17,18,19,20]])
    
    return [s1_img, s2_data, s3_lstm, s4_gru, s5_bi, s6_td, s7_text, s8_trans, s9_cnn_s, s10_cnn_d, s11_ae, s12_fnn1, s13_fnn2, s14_vae, s15_rbm, s16_graph, s17_attn, s18_cust, s19_res_f, s20_res_b, s21_res_final], [snn_vigilance, symbolic_bias, task_vector, retrieved_mem], ventral, dorsal, rsa_weights

# Cell 8 — Fusion, Symplectic Physics, and Output Heads
def finalize_tcs26(inputs_list, controls_list, ventral, dorsal, rsa_weights):
    snn_vigilance, symbolic_bias, task_vector, retrieved_mem = controls_list
    retrieved_state = retrieved_mem[:, :FRONTAL_LOBE_UNITS]
    retrieved_causal = retrieved_mem[:, FRONTAL_LOBE_UNITS:]

    # Fusion
    v_proj = GPlasticDense(FRONTAL_LOBE_UNITS)(ventral)
    d_proj = GPlasticDense(FRONTAL_LOBE_UNITS)(dorsal)
    fusion = BatchNormalization()(GPlasticDense(FRONTAL_LOBE_UNITS)(Concatenate()([v_proj, d_proj])))
    
    # Merge with Memory
    fused_mem = Concatenate()([fusion, retrieved_state])
    
    # Recurrence
    gru_out, wm_state = GPlasticGRU(FRONTAL_LOBE_UNITS)(tf.expand_dims(fused_mem, axis=1))

    # --- NEW: Symplectic Physics Stream ---
    # We project working memory into Phase Space (q, p) and evolve it
    symplectic_input = GPlasticDense(SYMPLECTIC_DIM, activation='tanh', name='to_phase_space')(wm_state)
    symplectic_evolved = SymplecticIntegratorLayer(SYMPLECTIC_DIM, name='symplectic_physics_engine')(symplectic_input)
    
    # HLS
    hls_raw = HLS_Processor()(wm_state)
    hls_dec = HLS_Decoder()(hls_raw)
    
    # Control Modules
    mcc_budget = GPlasticDense(1, activation='sigmoid')(wm_state)
    mcc_conf = GPlasticDense(1, activation='sigmoid')(wm_state)
    mlc_out = MetaLearningControl()(wm_state)
    bg_mask = BasalGangliaSelectionLayer(NUM_PFC_CONTEXTS)(wm_state, task_vector)
    axiom = AxiomaticKnowledgeLayer()(bg_mask)
    
    # CGW (Now includes Symplectic data)
    cgw_inputs = [gru_out, Multiply()([symbolic_bias, mcc_budget]), Multiply()([snn_vigilance, mcc_budget]), mcc_conf, hls_dec, axiom, symplectic_evolved]
    cgw_out = CGWAttentionLayer()(cgw_inputs)
    
    # PFC
    pfc_out = MultiContextExecutiveGating(FRONTAL_LOBE_UNITS, NUM_PFC_CONTEXTS)(cgw_out, bg_mask)
    
    # Outputs
    out_cls = GPlasticDense(NUM_CLASSES, activation='softmax')(pfc_out)
    out_state = GPlasticDense(FRONTAL_LOBE_UNITS*2 + CAUSAL_STATE_DIM, activation='sigmoid')(pfc_out)
    out_rew = GPlasticDense(1, activation='tanh')(pfc_out)
    out_causal = CausalInferenceModule()(pfc_out, retrieved_causal)
    
    all_inputs = inputs_list + controls_list
    outputs = [out_cls, out_state, out_rew, bg_mask, mcc_conf, rsa_weights, out_causal, hls_raw, mlc_out[:, 0:1], axiom, symplectic_evolved]
    
    return Model(inputs=all_inputs, outputs=outputs, name='TCS26_Symplectic_Omni')
