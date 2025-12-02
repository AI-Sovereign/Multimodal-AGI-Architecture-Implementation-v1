import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, Flatten, Embedding, LSTM, GRU, Bidirectional, TimeDistributed, GlobalMaxPooling1D, Concatenate, Multiply, BatchNormalization, Dot, Reshape, Dropout, Attention
from tensorflow.keras.models import Model
import numpy as np
import random
from config_consts import *

# --- Cell 2: G-Plasticity mixin and plastic layers ---
class GPlasticityMixin:
    """TCS-25's base mixin for Generalized Plasticity (G-Plasticity), now with dynamic rate."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Neuromodulatory signals (tf tensors)
        self.surprisal_signal = tf.constant([[0.0]], dtype=tf.float32)
        self.causal_signal = tf.constant([[0.0]], dtype=tf.float32)    
        self.dynamic_plasticity_strength = tf.constant([[0.0001]], dtype=tf.float32)
        self.w_old = None
        self.x_input = None
        self.y_output = None

    def calculate_plasticity_change(self):
        if self.w_old is None or self.x_input is None or self.y_output is None:
            return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

        hebbian_term = tf.matmul(tf.transpose(self.x_input), self.y_output) / tf.cast(tf.shape(self.x_input)[0], tf.float32)
        modulation_scalar = 0.6 * tf.cast(self.surprisal_signal[0, 0], tf.float32) + 0.4 * tf.cast(self.causal_signal[0, 0], tf.float32)
        delta_p = tf.cast(self.dynamic_plasticity_strength[0, 0], tf.float32) * modulation_scalar * hebbian_term
        return delta_p, tf.norm(delta_p)

    def track_activations(self, x_input, y_output):
        self.x_input = tf.stop_gradient(x_input)
        self.y_output = tf.stop_gradient(y_output)

class GPlasticDense(GPlasticityMixin, Dense):
    """G-Plastic Dense"""
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(units=units, activation=activation, **kwargs)

    def call(self, inputs):
        self.w_old = self.weights[0] if self.weights else None
        output = super().call(inputs)
        self.track_activations(inputs, output)
        return output

class GPlasticConv2D(GPlasticityMixin, Conv2D):
    """G-Plastic Conv2D"""
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, activation=activation, **kwargs)

    def call(self, inputs):
        self.w_old = self.weights[0] if self.weights else None
        output = super().call(inputs)
        flat_out = Flatten()(output)
        self.track_activations(inputs, flat_out)
        return output

class GPlasticGRU(GPlasticityMixin, Layer):
    """G-Plastic GRU wrapper"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.gru_cell = GRU(units, return_sequences=False, return_state=True)
        self.proj_input = GPlasticDense(units,    activation='tanh', name='gplastic_gru_input_proj')
        
    def call(self, inputs, initial_state=None):
        projected_input = self.proj_input(inputs) 
        output, state = self.gru_cell(projected_input, initial_state=initial_state)
        # store weight snapshot for potential plasticity usage
        self.w_old = self.gru_cell.weights[0] if self.gru_cell.weights else None
        return output, state

# --- Cell 3: AKL and EpisodicRelationalMemory ---
class AxiomaticKnowledgeLayer(Layer):
    """Provides an axiomatic, structured knowledge vector for context."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.axiom_embeddings = self.add_weight(
            name="axiom_emb_matrix",
            shape=(NUM_PFC_CONTEXTS, AXIOMATIC_DIM),
            initializer='glorot_uniform',
            trainable=True
        )
        self.context_to_axiom_proj = GPlasticDense(AXIOMATIC_DIM, activation='tanh', name='axiom_proj_plastic')

    def call(self, context_mask):
        expanded_mask = tf.expand_dims(context_mask, axis=-1)
        weighted_axioms = expanded_mask * self.axiom_embeddings
        return tf.reduce_sum(weighted_axioms, axis=1)

class EpisodicRelationalMemory:
    """Graph-based memory for episodes."""
    def __init__(self):
        self.buffer = []
        self.max_size = 10000
        self.graph_nodes = {}
        self.graph_edges = {}

    def store(self, state_vector, context_mask, causal_vector):
        node_id = len(self.buffer)
        episode = {
            'state': state_vector[0].numpy(),
            'context': context_mask[0].numpy(),
            'causal': causal_vector[0].numpy(),
            'id': node_id
        }
        self.buffer.append(episode)
        self.graph_nodes[node_id] = episode

        if node_id > 0:
            past_node_id = random.randint(0, node_id - 1)
            relation_vec = np.random.rand(CAUSAL_STATE_DIM).astype(np.float32)
            self.graph_edges.setdefault(past_node_id, []).append((node_id, relation_vec))

        if len(self.buffer) > self.max_size:
            old_node = self.buffer.pop(0)
            del self.graph_nodes[old_node['id']]
            if old_node['id'] in self.graph_edges:
                del self.graph_edges[old_node['id']]

    def retrieve_context(self, query_vector):
        if not self.graph_nodes:
            return tf.zeros((1, FRONTAL_LOBE_UNITS + CAUSAL_STATE_DIM), dtype=tf.float32)
        query = np.asarray(query_vector[0])
        query_len = query.shape[-1]
        scores = []
        nodes = list(self.graph_nodes.values())
        for node in nodes:
            node_state_slice = node['state'][:query_len]
            denom = (np.linalg.norm(query) * np.linalg.norm(node_state_slice) + 1e-8)
            score = float(np.dot(query, node_state_slice) / denom)
            scores.append(score)
        weights = np.exp(scores) / np.sum(np.exp(scores))
        retrieved_state = np.zeros(FRONTAL_LOBE_UNITS, dtype=np.float32)
        retrieved_causal = np.zeros(CAUSAL_STATE_DIM, dtype=np.float32)
        for i, node in enumerate(nodes):
            retrieved_state += weights[i] * node['state'][:FRONTAL_LOBE_UNITS]
            retrieved_causal += weights[i] * node['causal']
        retrieved_state = tf.constant(retrieved_state, dtype=tf.float32)[tf.newaxis, :]
        retrieved_causal = tf.constant(retrieved_causal, dtype=tf.float32)[tf.newaxis, :]
        return Concatenate()([retrieved_state, retrieved_causal])

# --- Cell 4: Relational Self-Attention and HTSP ---
class RelationalSelfAttention(Layer):
    """Dynamically scales features based on global context."""
    def __init__(self, num_features, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.relational_embeddings = self.add_weight(
            name="relational_emb_matrix",
            shape=(num_features, num_features, RELATIONAL_EMB_DIM),
               initializer='glorot_uniform',
            trainable=True
        )
        self.query_gen = GPlasticDense(feature_dim, activation='relu', name='rsa_query_plastic')
        self.key_gen = GPlasticDense(feature_dim, activation='relu', name='rsa_key_plastic')
        self.rel_context_proj_gate = GPlasticDense(self.num_features * self.num_features, activation='tanh', name='rel_context_proj_gate')
        self.attention_weights_generator = GPlasticDense(num_features, activation='softmax', name='rsa_weights_plastic')

    def call(self, projected_feature_list):
        global_context_fused = Concatenate()(projected_feature_list)
        queries = tf.stack([self.query_gen(f) for f in projected_feature_list], axis=1)
        keys = tf.stack([self.key_gen(f) for f in projected_feature_list], axis=1)
        attention_scores = Dot(axes=-1)([queries, keys])
        relational_context_proj = self.rel_context_proj_gate(global_context_fused)
        combined_context = Concatenate()([global_context_fused, relational_context_proj])
        attention_weights = self.attention_weights_generator(combined_context)
        attended_features = []
        for i, feature in enumerate(projected_feature_list):
            weight = attention_weights[:, i:i+1]
            weighted_feature = Multiply()([feature, weight])
            attended_features.append(weighted_feature)
        return attended_features, attention_weights, tf.reduce_mean(attention_scores)

class HTSP_Unit(Layer):
    """Manages temporal context across different time scales."""
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

# --- Cell 5: CGW and executive modules ---
class CGWAttentionLayer(Layer):
    """Conscious Global Workspace (CGW) Layer (TCS-25)."""
    def build(self, input_shape):
        # compute fusion dim
        fusion_dim = (input_shape[0][-1] + 1 + 1 + 1 + input_shape[4][-1] + input_shape[5][-1])
        self.kernel = self.add_weight(name="kernel_fusion", shape=(fusion_dim, input_shape[0][-1]), initializer='glorot_uniform', trainable=True)
        self.gating_dense = GPlasticDense(1, activation='sigmoid', name='cgw_gating_plastic')
        super().build(input_shape)
    def call(self, plastic_gru_output, scaled_symbolic_bias, scaled_vigilance, mcc_confidence, hls_vector, axiomatic_vector):
        fused = Concatenate()([plastic_gru_output, scaled_symbolic_bias, scaled_vigilance, mcc_confidence, hls_vector, axiomatic_vector])
        gated_fused = Multiply()([fused, self.gating_dense(fused)])
        output = tf.matmul(gated_fused, self.kernel)
        return output

class MetaLearningControl(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_pred = GPlasticDense(MLC_OUTPUT_DIM, activation='sigmoid', name='mlc_output_gplastic')
        self.dropout = Dropout(0.2)
    def call(self, pfc_gated_output):
        x = self.dropout(pfc_gated_output)
        return self.dense_pred(x)

class BasalGangliaSelectionLayer(Layer):
    def __init__(self, num_contexts, **kwargs):
        super().__init__(**kwargs)
        self.context_gate = GPlasticDense(num_contexts, activation='softmax', name='bg_context_gate_gplastic')
    def call(self, cgw_output, task_vector_input):
        fusion =    Concatenate()([cgw_output, task_vector_input])
        return self.context_gate(fusion)

class MultiContextExecutiveGating(Layer):
    def __init__(self, units, num_contexts, **kwargs):
        super().__init__(**kwargs)
        self.context_networks = [GPlasticDense(units, activation='tanh', name=f'pfc_context_gplastic_{i}') for i in range(num_contexts)]
    def call(self, cgw_output, context_mask):
        context_outputs = tf.stack([net(cgw_output) for net in self.context_networks], axis=1)
        weighted_mask = tf.expand_dims(context_mask, axis=-1)
        gated_output = Multiply()([context_outputs, weighted_mask])
        return tf.reduce_sum(gated_output, axis=1)

class CausalInferenceModule(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = GPlasticDense(CAUSAL_STATE_DIM, activation='sigmoid', name='cim_causal_output_gplastic')
        self.gating = GPlasticDense(CAUSAL_STATE_DIM, activation='tanh', name='cim_gating_gplastic')
    def call(self, pfc_gated_output, retrieved_causal_context):
        fused = Concatenate()([pfc_gated_output, retrieved_causal_context])
        return Multiply()([self.dense(fused), self.gating(fused)])

# --- Cell 6: HLS Processor ---
class HLS_Processor(Layer):
    """Projects the PFC output into the massive, sparse HLS."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_proj = GPlasticDense(FRONTAL_LOBE_UNITS * 2, activation='relu', name='hls_input_proj_plastic')
        # Minimal fix: instantiate as a plastic dense layer (previous mixin misuse corrected)
        self.latent_dense = GPlasticDense(HYPER_LATENT_DIM, activation='relu', name='hls_core_plastic', use_bias=False)
    def build(self, input_shape):
        self.latent_dense.build(input_shape)
        super().build(input_shape)
    def call(self, pfc_output):
        x = self.input_proj(pfc_output)
        hls_output_raw = self.latent_dense(x)
        return hls_output_raw

class HLS_Decoder(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = GPlasticDense(FRONTAL_LOBE_UNITS // 2, activation='tanh', name='hls_decoder_plastic')
    def call(self, hls_vector):
        return self.decoder(hls_vector)

# --- Cells 7, 8, 9: Core Model Builder ---
def build_tcs_net_core_model(image_shape, data_size, ts_steps, ts_dim, seq_len, seq_dim, graph_dim, num_classes):
    # Inputs
    s1_image_input = Input(shape=image_shape, name='stream_01_image_input')
    s9_small_cnn_input = Input(shape=(32, 32, 1), name='stream_09_small_cnn_input')
    s10_deep_cnn_input = Input(shape=(64, 64, 3), name='stream_10_deep_cnn_input')
    s11_conv_ae_input = Input(shape=(16, 16, 8), name='stream_11_conv_autoencoder_input')
    s3_lstm_input = Input(shape=(ts_steps, ts_dim), name='stream_03_lstm_input')
    s4_gru_input = Input(shape=(ts_steps, ts_dim), name='stream_04_gru_input')
    s5_bidirectional_input = Input(shape=(ts_steps, ts_dim), name='stream_05_bidirectional_input')
    s6_timedistributed_input = Input(shape=(ts_steps, ts_dim), name='stream_06_timedistributed_input')
    s7_text_seq_input = Input(shape=(seq_len,), name='stream_07_text_seq_input')
    s8_transformer_input = Input(shape=(seq_len, seq_dim), name='stream_08_transformer_input')
    s2_structured_data_input = Input(shape=(data_size,), name='stream_02_structured_data_input')
    s12_fnn_input_100 = Input(shape=(100,), name='stream_12_fnn_input_100')
    s13_fnn_input_256 = Input(shape=(256,), name='stream_13_fnn_input_256')
    s14_vae_latent_input = Input(shape=(128,), name='stream_14_vae_latent_input')
    s15_rbm_feature_input = Input(shape=(64,), name='stream_15_rbm_feature_input')
    s16_graph_input_flat = Input(shape=(graph_dim * graph_dim,), name='stream_16_graph_input_flat')
    s17_attention_vector_input = Input(shape=(40,), name='stream_17_attention_vector_input')
    s18_custom_encoder_input = Input(shape=(80,),    name='stream_18_custom_encoder_input')
    s19_residual_fwd_input = Input(shape=(512,), name='stream_19_residual_fwd_input')
    s20_residual_bwd_input = Input(shape=(512,), name='stream_20_residual_bwd_input')
    s21_residual_final_input = Input(shape=(512,), name='stream_21_residual_final_input')
    snn_vigilance_input = Input(shape=(1,), name='snn_vigilance_input')
    symbolic_bias_input = Input(shape=(1,), name='symbolic_bias_input')
    task_vector_input = Input(shape=(16,), name='task_vector_input')

    retrieved_memory_and_causal = Input(shape=(FRONTAL_LOBE_UNITS + CAUSAL_STATE_DIM,), name='erm_retrieval_input')
    retrieved_state  = retrieved_memory_and_causal[:, :FRONTAL_LOBE_UNITS]
    retrieved_causal_context = retrieved_memory_and_causal[:, FRONTAL_LOBE_UNITS:]

    input_list = [s1_image_input, s2_structured_data_input, s3_lstm_input, s4_gru_input, s5_bidirectional_input,
                 s6_timedistributed_input, s7_text_seq_input, s8_transformer_input, s9_small_cnn_input,
                 s10_deep_cnn_input, s11_conv_ae_input, s12_fnn_input_100, s13_fnn_input_256,
                 s14_vae_latent_input, s15_rbm_feature_input, s16_graph_input_flat, s17_attention_vector_input,
                 s18_custom_encoder_input, s19_residual_fwd_input, s20_residual_bwd_input, s21_residual_final_input]
    all_inputs = input_list + [snn_vigilance_input, symbolic_bias_input, task_vector_input, retrieved_memory_and_causal]

    # Helper
    def extract_and_hybrid(input_tensor, units, name):
        return GPlasticDense(units, activation='tanh', name=f'hybrid_synapse_gplastic_{name}')(input_tensor)

    # Feature extraction
    x = GPlasticConv2D(128, 3, activation='relu', padding='same')(s1_image_input)
    f1 = extract_and_hybrid(Flatten()(x), 512, 'image')
    s7_embedded = Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=SEQ_LEN)(s7_text_seq_input)
    f7 = extract_and_hybrid(GlobalMaxPooling1D()(s7_embedded), 256, 'text')
    f3 = extract_and_hybrid(LSTM(256)(s3_lstm_input), 512, 'lstm')
    f4 = extract_and_hybrid(GRU(256)(s4_gru_input), 512, 'gru')
    f5 = extract_and_hybrid(Bidirectional(LSTM(256))(s5_bidirectional_input), 512, 'bidir')
    s6_td_output = TimeDistributed(GPlasticDense(128, activation='relu'))(s6_timedistributed_input)
    f6 = extract_and_hybrid(GlobalMaxPooling1D()(s6_td_output), 256, 'timedist')
    f2 = extract_and_hybrid(s2_structured_data_input, 256, 'structured')
    f8 = extract_and_hybrid(Flatten()(s8_transformer_input), 512, 'transformer')
    f9_conv = GPlasticConv2D(64, 3, activation='relu')(s9_small_cnn_input)
    f9 = extract_and_hybrid(Flatten()(f9_conv), 256, 'small_cnn')
    f10_conv = GPlasticConv2D(128, 3, activation='relu')(s10_deep_cnn_input)
    f10 = extract_and_hybrid(Flatten()(f10_conv), 512, 'deep_cnn')
    f11_conv = GPlasticConv2D(64, 3, activation='relu')(s11_conv_ae_input)
    f11 = extract_and_hybrid(Flatten()(f11_conv), 256, 'conv_ae')
    f12 = extract_and_hybrid(s12_fnn_input_100, 256, 'fnn_100')
    f13 = extract_and_hybrid(s13_fnn_input_256, 512, 'fnn_256')
    f14 = extract_and_hybrid(s14_vae_latent_input, 512, 'vae_latent')
    f15 = extract_and_hybrid(s15_rbm_feature_input, 256, 'rbm_feature')
    f16 = extract_and_hybrid(s16_graph_input_flat, 512, 'graph_flat')
    f17 = extract_and_hybrid(s17_attention_vector_input, 256, 'attention_vec')
    f18 = extract_and_hybrid(s18_custom_encoder_input, 512, 'custom_encoder')
    f19 = extract_and_hybrid(s19_residual_fwd_input, 512, 'residual_fwd')
    f20 = extract_and_hybrid(s20_residual_bwd_input, 512, 'residual_bwd')
    f21 = extract_and_hybrid(s21_residual_final_input, 512, 'residual_final')

    all_features_raw = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21]

    # RSA projections
    PROJECTED_FEATURE_DIM = 512
    projected_features = [GPlasticDense(PROJECTED_FEATURE_DIM, activation='relu', name=f'rsa_proj_gplastic_{i}')(f) for i, f in    enumerate(all_features_raw)]

    rsa_features, rsa_weights, rsa_scores = RelationalSelfAttention(num_features=21, feature_dim=PROJECTED_FEATURE_DIM, name='relational_self_attention_unit')(projected_features)

    ventral_indices = [0, 6, 7, 8, 9, 10, 1, 11, 12, 13, 14, 16]
    dorsal_indices = [2, 3, 4, 5, 15, 17, 18, 19, 20]

    rsa_ventral = [rsa_features[i] for i in ventral_indices]
    rsa_dorsal = [rsa_features[i] for i in dorsal_indices]

    ventral_features = Concatenate(name='ventral_path_fusion_rsa')(rsa_ventral)
    ventral_features = GPlasticDense(FRONTAL_LOBE_UNITS, name='ventral_synapse_gplastic')(ventral_features)

    dorsal_features = Concatenate(name='dorsal_path_fusion_rsa')(rsa_dorsal)
    dorsal_features = GPlasticDense(FRONTAL_LOBE_UNITS, name='dorsal_synapse_gplastic')(dorsal_features)

    fusion_features = Concatenate(name='core_fusion')([ventral_features, dorsal_features])
    fusion_features = GPlasticDense(FRONTAL_LOBE_UNITS, name='core_pfc_input_synapse')(fusion_features)
    fusion_features = BatchNormalization(name='pfc_input_bn')(fusion_features)

    fused_with_memory = Concatenate(name='fusion_with_erm')([fusion_features, retrieved_state])

    # HTSP placeholders (kept as Inputs so model signature matches your design)
    htsp_fast_state = Input(shape=(FRONTAL_LOBE_UNITS,), name='htsp_fast_state')
    htsp_slow_state_h = Input(shape=(FRONTAL_LOBE_UNITS,), name='htsp_slow_state_h')
    htsp_slow_state_c = Input(shape=(FRONTAL_LOBE_UNITS,), name='htsp_slow_state_c')

    # GRU recurrence (plastic)
    plastic_gru_output, current_working_memory_state = GPlasticGRU(FRONTAL_LOBE_UNITS, name='erm_gplastic_gru')(tf.expand_dims(fused_with_memory, axis=1))

    # HLS injection
    hls_vector_raw = HLS_Processor(name='hls_processor')(current_working_memory_state)
    hls_decoded = HLS_Decoder(name='hls_decoder')(hls_vector_raw)

    # Meta-Cognitive Control (MCC)
    mcc_attention_budget = GPlasticDense(1, activation='sigmoid', name='mcc_attention_budget_gplastic')(current_working_memory_state)
    mcc_confidence = GPlasticDense(1, activation='sigmoid', name='mcc_confidence_gplastic')(current_working_memory_state)

    # Meta-Learning Control (MLC)
    mlc_output = MetaLearningControl(name='meta_learning_control_head')(current_working_memory_state)
    mlc_plasticity_rate = mlc_output[:, 0:1]

    # Basal Ganglia (BG) and Axiomatic Knowledge Integration
    bg_context_mask = BasalGangliaSelectionLayer(NUM_PFC_CONTEXTS, name='basal_ganglia_selector')(
        current_working_memory_state, task_vector_input
    )
    axiomatic_knowledge_vector = AxiomaticKnowledgeLayer(name='axiomatic_knowledge_layer')(bg_context_mask)

    # Conscious Global Workspace (CGW)
    scaled_symbolic_bias = Multiply()([symbolic_bias_input, mcc_attention_budget])
    scaled_vigilance = Multiply()([snn_vigilance_input, mcc_attention_budget])

    cgw_output = CGWAttentionLayer(name='conscious_global_workspace')(
        plastic_gru_output,
        scaled_symbolic_bias,
        scaled_vigilance,
        mcc_confidence,
        hls_decoded,
        axiomatic_knowledge_vector
    )

    # Prefrontal Cortex (PFC) Executive Gating
    pfc_gated_output = MultiContextExecutiveGating(
        FRONTAL_LOBE_UNITS, NUM_PFC_CONTEXTS, name='executive_gating_pfc_output'
    )(cgw_output, bg_context_mask)

    # PWM Heads (Predictive Working Memory Outputs)
    PWM_STATE_DIM = FRONTAL_LOBE_UNITS * 2 + CAUSAL_STATE_DIM

    predicted_next_state = GPlasticDense(
        PWM_STATE_DIM, activation='sigmoid', name='pwm_next_state_prediction_gplastic'
    )(pfc_gated_output)

    predicted_next_reward = GPlasticDense(
        1, activation='tanh', name='pwm_next_reward_prediction_scalar_gplastic'
    )(pfc_gated_output)

    # Causal Inference Module
    predicted_causal_state = CausalInferenceModule(name='causal_inference_module')(
        pfc_gated_output, retrieved_causal_context
    )

    # Final Outputs
    final_classification_output =    GPlasticDense(
        NUM_CLASSES, activation='softmax', name='classification_output_gplastic'
    )(pfc_gated_output)

    final_confidence_output = mcc_confidence

    # --- Assemble Final Core Model ---
    tcs_net_core = Model(
        inputs=all_inputs,
        outputs=[
            final_classification_output,
            predicted_next_state,
            predicted_next_reward,
            bg_context_mask,
            final_confidence_output,
            rsa_weights,
            predicted_causal_state,
            hls_vector_raw,
            mlc_plasticity_rate,
            axiomatic_knowledge_vector,
        ],
        name='Temporal_Causal_Synthesis_Network_Core'
    )
    return tcs_net_core
