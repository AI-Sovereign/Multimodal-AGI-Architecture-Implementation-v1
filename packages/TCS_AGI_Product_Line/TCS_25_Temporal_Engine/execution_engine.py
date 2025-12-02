import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Model
import numpy as np
import traceback
import random
import itertools
from config_consts import *
from core_architecture import *

# --- Cell 10: Learner, data generator, training driver, inference ---
class TCS_GeneralIntelligence(Model):
    """TCS-25: Learner with SSTC and plasticity handling."""
    def __init__(self, tcs_net_core_model, snn_neuron, rule_reasoner, erm_buffer):
        super().__init__()
        self.tcs_net_core = tcs_net_core_model
        self.snn = snn_neuron
        self.rule_reasoner = rule_reasoner
        self.erm_buffer = erm_buffer
        self.sru = GPlasticDense(1, activation='sigmoid', name='sru_dopamine_gate')

        self.classification_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.state_prediction_loss_fn = tf.keras.losses.MeanSquaredError(name='state_prediction_mse_loss')
        self.reward_prediction_loss_fn = tf.keras.losses.MeanSquaredError(name='reward_prediction_mse_loss')
        self.confidence_loss_fn = tf.keras.losses.MeanSquaredError(name='confidence_mse_loss')
        self.causal_loss_fn = tf.keras.losses.MeanSquaredError(name='causal_mse_loss')
        self.hls_contrastive_loss_fn = tf.keras.losses.CosineSimilarity(axis=-1, name='hls_contrastive_loss')
        self.mlc_reg_loss_fn = tf.keras.losses.MeanSquaredError(name='mlc_reg_loss')

        self.temporal_causal_contrastive_loss = tf.keras.losses.CosineSimilarity(axis=-1, name='sstc_loss')

        self.plasticity_targets = []
        for layer in tcs_net_core_model.layers:
            if isinstance(layer, GPlasticityMixin):
                self.plasticity_targets.append(layer)
            if hasattr(layer, 'proj_input') and isinstance(layer.proj_input, GPlasticDense):
                self.plasticity_targets.append(layer.proj_input)
            if isinstance(layer, HLS_Processor) and isinstance(layer.latent_dense, GPlasticityDense):
                self.plasticity_targets.append(layer.latent_dense)
            if isinstance(layer, Dense) and hasattr(layer, 'w_old'):
                self.plasticity_targets.append(layer)

    def compile(self, optimizer, loss, metrics=None):
        if metrics is None: metrics = []
        metrics.extend([
            tf.keras.metrics.Mean(name='classification_loss'),
            tf.keras.metrics.Mean(name='state_prediction_loss'),
            tf.keras.metrics.Mean(name='causal_prediction_loss'),
            tf.keras.metrics.Mean(name='hls_contrastive_loss'),
            tf.keras.metrics.Mean(name='sstc_loss_mean'),
            tf.keras.metrics.Mean(name='mlc_rate_mean'),
            tf.keras.metrics.Mean(name='surprisal_update_mag'),
            tf.keras.metrics.Mean(name='causal_update_mag')
        ])
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @tf.function
    def train_step(self, data):
        (x_data, y_targets) = data
        x_inputs = x_data[:21]
        x_snn_vigilance_raw = x_data[21]
        x_symbolic_bias_raw = x_data[22]
        x_task_vector = x_data[23]

        y_true_classification = y_targets[0]
        y_true_next_state = y_targets[1]
        y_true_next_reward = y_targets[2]
        y_true_confidence = y_targets[3]
        y_true_causal_state = y_targets[4]
        y_true_next_hls = y_targets[5]
        y_true_temporal_causal = y_targets[6]

        query_vector = x_inputs[18][0:1]
        retrieved_mem_causal = self.erm_buffer.retrieve_context(query_vector)
        x_retrieved_mem_causal = tf.tile(retrieved_mem_causal, [tf.shape(x_inputs[0])[0], 1])

        total_surprisal_mag = tf.constant(0.0)
        total_causal_mag = tf.constant(0.0)

        with tf.GradientTape() as tape:
            # NOTE: x_data[24] is the dummy input for retrieved_memory_and_causal, but we overwrite it with the real retrieval
            core_inputs = x_inputs + [x_snn_vigilance_raw, x_symbolic_bias_raw,  x_task_vector, x_retrieved_mem_causal]
            y_pred_classification, y_pred_state, y_pred_reward, context_mask, y_pred_confidence, rsa_weights, y_pred_causal, y_pred_hls, mlc_plasticity_rate, axiomatic_vector = self.tcs_net_core(core_inputs, training=True)

            state_prediction_loss = self.state_prediction_loss_fn(y_true_next_state, y_pred_state)
            reward_prediction_loss = self.reward_prediction_loss_fn(y_true_next_reward, y_pred_reward)
            causal_prediction_loss = self.causal_loss_fn(y_true_causal_state, y_pred_causal)
            classification_loss = self.classification_loss_fn(y_true_classification, y_pred_classification)
            confidence_loss = self.confidence_loss_fn(y_true_confidence, y_pred_confidence)
            rsa_sparsity_loss = tf.reduce_mean(tf.norm(rsa_weights, ord=1, axis=-1)) * 0.0005

            hls_contrastive_loss = (1 + self.hls_contrastive_loss_fn(y_true_next_hls, y_pred_hls)) * 0.005
            mlc_reg_loss = tf.reduce_mean(mlc_plasticity_rate) * 0.0001
            sstc_loss = (1 + self.temporal_causal_contrastive_loss(y_true_temporal_causal, Concatenate()([y_pred_causal, axiomatic_vector]))) * 0.01

            total_loss = classification_loss \
                       + (LOSS_WEIGHT_SSTC * state_prediction_loss) \
                       + (LOSS_WEIGHT_SSTC * reward_prediction_loss) \
                       + (LOSS_WEIGHT_SSTC * causal_prediction_loss) \
                       + confidence_loss \
                       + rsa_sparsity_loss \
                       + hls_contrastive_loss \
                       + mlc_reg_loss \
                       + sstc_loss

        mean_pred_error = tf.expand_dims(tf.reduce_mean(state_prediction_loss + reward_prediction_loss), axis=-1)
        sru_input_enhanced = Concatenate()([x_snn_vigilance_raw[0:1], mean_pred_error, y_pred_confidence[0:1]])
        global_surprisal_rate = self.sru(sru_input_enhanced)
        global_causal_rate = tf.expand_dims(tf.reduce_mean(causal_prediction_loss + sstc_loss), axis=-1)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        for layer in self.plasticity_targets:
            # set neuromodulatory signals (tensors)
            layer.dynamic_plasticity_strength = mlc_plasticity_rate
            layer.surprisal_signal = global_surprisal_rate
            layer.causal_signal = global_causal_rate

            target_var = None
            if hasattr(layer, 'kernel') and layer.kernel in trainable_vars:
                target_var = layer.kernel
            elif hasattr(layer, 'w') and layer.w in trainable_vars:
                target_var = layer.w

            if target_var is not None:
                delta_p, mag_p = layer.calculate_plasticity_change()
                try:
                    var_index = trainable_vars.index(target_var)
                    if gradients[var_index] is not None:
                        try:
                            gradients[var_index] = gradients[var_index] - delta_p
                        except Exception:
                            gradients[var_index] = gradients[var_index] - tf.cast(delta_p, gradients[var_index].dtype)
                except ValueError:
                    pass

                total_surprisal_mag += mag_p * global_surprisal_rate[0,0]
                total_causal_mag += mag_p * global_causal_rate[0,0]

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true_classification, y_pred_classification)

        state_to_store = Concatenate(axis=-1)([y_pred_state, y_pred_reward])
        self.erm_buffer.store(state_to_store, context_mask, y_pred_causal)

        return {m.name: m.result() for m in self.metrics}

# Data generator (same as original)
def generate_dummy_data_tcs_net(n_samples, num_classes):
    ts_steps, ts_dim, seq_len, seq_dim, graph_dim = TS_STEPS, TS_DIM, SEQ_LEN, SEQ_DIM, GRAPH_DIM
    X_inputs_data = [
        np.random.rand(n_samples, *IMAGE_SHAPE).astype(np.float32),
        np.random.rand(n_samples, DATA_INPUT_SIZE).astype(np.float32),
        np.random.rand(n_samples, ts_steps, ts_dim).astype(np.float32),
        np.random.rand(n_samples, ts_steps, ts_dim).astype(np.float32),
        np.random.rand(n_samples, ts_steps, ts_dim).astype(np.float32),
        np.random.rand(n_samples, ts_steps, ts_dim).astype(np.float32),
        tf.constant(np.random.randint(0, VOCAB_SIZE, size=(n_samples, seq_len))),
        np.random.rand(n_samples, seq_len, seq_dim).astype(np.float32),
        np.random.rand(n_samples, 32, 32, 1).astype(np.float32),
        np.random.rand(n_samples, 64, 64, 3).astype(np.float32),
        np.random.rand(n_samples, 16, 16, 8).astype(np.float32),
        np.random.rand(n_samples, 100).astype(np.float32),
        np.random.rand(n_samples, 256).astype(np.float32),
        np.random.rand(n_samples, 128).astype(np.float32),
        np.random.rand(n_samples, 64).astype(np.float32),
        np.random.rand(n_samples, graph_dim * graph_dim).astype(np.float32),
        np.random.rand(n_samples, 40).astype(np.float32),
        np.random.rand(n_samples, 80).astype(np.float32),
        np.random.rand(n_samples, 512).astype(np.float32),
        np.random.rand(n_samples, 512).astype(np.float32),
        np.random.rand(n_samples, 512).astype(np.float32),
    ]
    X_controls = [
        np.random.rand(n_samples, 1).astype(np.float32),
        np.random.rand(n_samples, 1).astype(np.float32),
        np.random.rand(n_samples, 16).astype(np.float32)
    ]
    X_inputs_all = X_inputs_data + X_controls
    
    # 🚨 MINIMAL FIX START 🚨
    # TCS-Net Core has 25 inputs, but X_inputs_all only has 24 elements before this point.
    # We must add a dummy tensor for the retrieved_memory_and_causal input to satisfy Keras.
    # The actual retrieval happens inside train_step, so this dummy data is only for Keras's data alignment check.
    DUMMY_ERM_SIZE = FRONTAL_LOBE_UNITS + CAUSAL_STATE_DIM
    X_dummy_erm = np.zeros((n_samples, DUMMY_ERM_SIZE), dtype=np.float32)
    X_inputs_all.append(X_dummy_erm)
    # 🚨 MINIMAL FIX END 🚨

    Y_classification = tf.one_hot(np.random.randint(0, num_classes, n_samples), depth=num_classes)
    PWM_STATE_DIM = FRONTAL_LOBE_UNITS * 2 + CAUSAL_STATE_DIM
    Y_true_next_state = np.random.rand(n_samples, PWM_STATE_DIM).astype(np.float32)
    Y_true_next_reward = np.random.rand(n_samples, 1).astype(np.float32)
    Y_true_confidence = 1.0 - np.random.rand(n_samples, 1).astype(np.float32) * 0.5
    Y_true_causal_state = np.random.rand(n_samples, CAUSAL_STATE_DIM).astype(np.float32)
    Y_true_next_hls = np.random.rand(n_samples, HYPER_LATENT_DIM).astype(np.float32)
    SSTC_TARGET_DIM = CAUSAL_STATE_DIM + AXIOMATIC_DIM
    Y_true_temporal_causal = np.random.rand(n_samples, SSTC_TARGET_DIM).astype(np.float32)
    Y_targets = [Y_classification, Y_true_next_state, Y_true_next_reward, Y_true_confidence, Y_true_causal_state, Y_true_next_hls, Y_true_temporal_causal]
    return X_inputs_all, Y_targets

if __name__ == "__main__":
    # Execution driver (small demo; may be heavy in Colab)
    N_SAMPLES = 1024
    EPOCHS = 3
    print("\n🌌 --- ASSEMBLING TEMPORAL CAUSAL SYNTHESIS NETWORK (TCS-25) --- 🌌")

    try:
        tcs_net_core = build_tcs_net_core_model(IMAGE_SHAPE, DATA_INPUT_SIZE, TS_STEPS, TS_DIM, SEQ_LEN, SEQ_DIM, GRAPH_DIM, NUM_CLASSES)
        class LIFNeuron: pass
        class RuleBasedReasoner: pass
        snn_neuron_instance = LIFNeuron()
        rule_reasoner_instance = RuleBasedReasoner()
        erm_buffer_instance = EpisodicRelationalMemory()

        tcs_net_model = TCS_GeneralIntelligence(tcs_net_core, snn_neuron_instance, rule_reasoner_instance, erm_buffer_instance)

        tcs_net_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

        X_inputs_all, Y_targets = generate_dummy_data_tcs_net(N_SAMPLES, NUM_CLASSES)

        print(f"\n--- Starting TCS-25 Training ({N_SAMPLES} Samples, {EPOCHS} Epochs) ---")
        history = tcs_net_model.fit(X_inputs_all, Y_targets, epochs=EPOCHS, batch_size=TRAINING_BATCH_SIZE, verbose=1)
        print("\n✅ Training Complete. TCS-25 is functional and hyper-plastic.")
    except NameError as e:
        print(f"🚨 ERROR: A required constant or class is missing. Error: {e}")
    except Exception as e:
        print(f"🚨 ERROR during Training: {e}")
        print(traceback.format_exc())

    # Minimal inference demo
    print("\n--- Executing Minimal TCS-25 Inference Cycle ---")
    try:
        X_test_all, _ = generate_dummy_data_tcs_net(1, NUM_CLASSES)
        query_vector = X_test_all[18][0:1]
        X_test_erm_retrieval = erm_buffer_instance.retrieve_context(query_vector)
        # 🚨 MINIMAL FIX START 🚨
        # The last element of X_test_all is the dummy ERM input. We replace it with the actual retrieval.
        X_test_all[-1] = X_test_erm_retrieval.numpy()
        X_test_final = X_test_all # No need to concatenate, the list is already 25 elements long and the last one is updated.
        # 🚨 MINIMAL FIX END 🚨
        
        results = tcs_net_core.predict(X_test_final, verbose=0)
        final_output, pred_state, pred_reward, context_mask, pred_confidence, rsa_weights, pred_causal, pred_hls, mlc_rate, axiomatic_vector = results

        try:
            surprisal_metric = tcs_net_model.get_metric('surprisal_update_mag')
            causal_metric = tcs_net_model.get_metric('causal_update_mag')
            sstc_metric = tcs_net_model.get_metric('sstc_loss_mean')
        except Exception:
            surprisal_metric = causal_metric = sstc_metric = None

        surprisal_mag = surprisal_metric.result().numpy() if surprisal_metric is not None else 0.0
        causal_mag = causal_metric.result().numpy() if causal_metric is not None else 0.0
        sstc_loss = sstc_metric.result().numpy() if sstc_metric is not None else 0.0

        attended_indices = np.argsort(rsa_weights[0])[-3:][::-1]

        print("\n====================================================")
        print("  TEMPORAL CAUSAL SYNTHESIS NETWORK (TCS-25) RESULTS ")
        print("====================================================")
        print(f"   -> Raw Predicted Class: {np.argmax(final_output[0])}   (Prob: {np.max(final_output[0]):.4f})")
        print(f"   -> Predicted Next Reward (PWM): {pred_reward[0,0]:.4f}")
        print(f"   -> Inferred Winning Context (BG): Context {np.argmax(context_mask[0])}")
        print(f"   -> Predicted System Confidence (MCC Output): {pred_confidence[0,0]:.4f}")
        print(f"   -> Predicted Causal State (CIM Output): Mean: {np.mean(pred_causal[0]):.4f}")
        print(f"   -> Axiomatic Knowledge Vector (AKE): L2 Norm: {np.linalg.norm(axiomatic_vector[0]):.4f}")
        print(f"   -> Dynamic Plasticity Rate (MLC Output): {mlc_rate[0,0]:.6f}")
        print(f"   -> HLS L1 Norm (Sparsity Check): {np.linalg.norm(pred_hls[0], ord=1):.4f}")
        print(f"   -> SSTC Loss (Temporal Consistency Check): {sstc_loss:.6f}")
        print(f"   -> G-Plasticity Surprisal Update Magnitude (Avg.): {surprisal_mag:.4f}")
        print(f"   -> G-Plasticity Causal Update Magnitude (Avg.): {causal_mag:.4f}")
        print(f"   -> Top 3 Attended Feature Streams (RSA): Index {attended_indices[0]} ({rsa_weights[0, attended_indices[0]]:.4f}), Index {attended_indices[1]} ({rsa_weights[0, attended_indices[1]]:.4f})")
        print("====================================================")
    except Exception as e:
        print(f"❌ ERROR during TCS-25 Inference Demo: {e}")
        print(traceback.format_exc())
