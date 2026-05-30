[+] Successfully wrote global sitecustomize.py to /usr/local/lib/python3.12/dist-packages/sitecustomize.py
2026-05-30 14:46:54.524859: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO 05-30 14:47:04 __init__.py:190] Automatically detected platform cuda.
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth: Could not find `steps_per_generation` in grpo_trainer
Unsloth: Could not find `generation_batch_size` in grpo_trainer
[+] Core machine learning environments detected.
[+] transformers quantizers/auto.py torchao guard already applied.
[+] vLLM inputs/registry.py already patched.
[+] vLLM multimodal/processing.py already patched.
[*] Cleared existing unsloth_compiled_cache for clean regeneration.
Unsloth: UnslothAlignPropTrainer is already patched.
Unsloth: UnslothBCOTrainer is already patched.
Unsloth: UnslothCPOTrainer is already patched.
Unsloth: UnslothDDPOTrainer is already patched.
Unsloth: UnslothDPOTrainer is already patched.
Unsloth: UnslothGKDTrainer is already patched.
Unsloth: Could not find `steps_per_generation` in grpo_trainer
Unsloth: Could not find `generation_batch_size` in grpo_trainer
Unsloth: UnslothKTOTrainer is already patched.
Unsloth: UnslothNashMDTrainer is already patched.
Unsloth: UnslothOnlineDPOTrainer is already patched.
Unsloth: UnslothORPOTrainer is already patched.
Unsloth: UnslothPPOTrainer is already patched.
Unsloth: UnslothPRMTrainer is already patched.
Unsloth: UnslothRewardTrainer is already patched.
Unsloth: UnslothRLOOTrainer is already patched.
Unsloth: UnslothSFTTrainer is already patched.
Unsloth: UnslothXPOTrainer is already patched.
[*] Initializing compiler for Unsloth GRPOTrainer...
[-] Error: Compiled trainer file not found at the expected path.
[+] GRPOConfig and GRPOTrainer successfully loaded!
[*] Starting LF-GRPO training pipeline in mode: STEP-GRPO
[*] Pre-processing GSM8K dataset...
[+] Loaded 7473 training samples and saved to data/gsm8k_train_grpo.jsonl
[+] vLLM 0.7.2 probe passed — will use vLLM for generation.
[+] Hardware detected: CUDA. Using vLLM for generation: True
[*] Loading model unsloth/Qwen2.5-1.5B-Instruct in 4-bit...
==((====))==  Unsloth 2026.5.8: Fast Qwen2 patching. Transformers: 4.57.6. vLLM: 0.7.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.5.1+cu124. CUDA: 7.5. CUDA Toolkit: 12.4. Triton: 3.1.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.28.post3. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit does not have a padding token! Will use pad_token = <|PAD_TOKEN|>.
[+] Spatial configuration: Freezing L0-L23. Adapting periphery layers: [24, 25, 26, 27]
Not an error, but Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters
are not enabled or a bias term (like in Qwen) is used.
Not an error, but Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters
are not enabled or a bias term (like in Qwen) is used.
Unsloth 2026.5.8 patched 28 layers with 28 QKV layers, 4 O layers and 4 MLP layers.
[*] Loading adapter weights from /content/grpo_cot_output...
[+] Adapter weights loaded — stage-1 weights restored, continuing from step 100.
Generating train split: 7473 examples [00:00, 131261.37 examples/s]
[*] GRPOConfig has no 'vllm_mode' param — skipping.
[*] GRPOConfig has no 'vllm_enforce_eager' param — skipping.
[*] vLLM fp16 inference model: Qwen/Qwen2.5-1.5B-Instruct
[+] Drive already mounted → /content/drive/MyDrive/grpo_checkpoints/grpo_cot_resumed
[*] vLLM: overriding model 'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit' → 'Qwen/Qwen2.5-1.5B-Instruct'
`torch_dtype` is deprecated! Use `dtype` instead!
WARNING 05-30 14:47:50 config.py:2386] Casting torch.bfloat16 to torch.float16.
INFO 05-30 14:48:06 config.py:542] This model supports multiple tasks: {'score', 'reward', 'embed', 'classify', 'generate'}. Defaulting to 'generate'.
INFO 05-30 14:48:06 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=Qwen/Qwen2.5-1.5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False, 
INFO 05-30 14:48:07 cuda.py:179] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 05-30 14:48:07 cuda.py:227] Using XFormers backend.
INFO 05-30 14:48:08 model_runner.py:1110] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
INFO 05-30 14:48:09 weight_utils.py:252] Using model weights format ['*.safetensors']
INFO 05-30 14:48:09 weight_utils.py:297] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards: 100% 1/1 [00:12<00:00, 12.52s/it]
INFO 05-30 14:48:22 model_runner.py:1115] Loading model weights took 2.8875 GB
INFO 05-30 14:48:28 worker.py:267] Memory profiling takes 5.75 seconds
INFO 05-30 14:48:28 worker.py:267] the current vLLM instance can use total_gpu_memory (14.56GiB) x gpu_memory_utilization (0.50) = 7.28GiB
INFO 05-30 14:48:28 worker.py:267] model weights take 2.89GiB; non_torch_memory takes 0.01GiB; PyTorch activation peak memory takes 2.02GiB; the rest of the memory reserved for KV Cache is 2.36GiB.
INFO 05-30 14:48:29 executor_base.py:110] # CUDA blocks: 5519, # CPU blocks: 9362
INFO 05-30 14:48:29 executor_base.py:115] Maximum concurrency for 32768 tokens per request: 2.69x
INFO 05-30 14:48:34 model_runner.py:1434] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100% 35/35 [00:45<00:00,  1.30s/it]
INFO 05-30 14:49:20 model_runner.py:1562] Graph capturing finished in 46 secs, took 0.17 GiB
INFO 05-30 14:49:20 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 57.90 seconds
Fetching 1 files: 100% 1/1 [00:00<00:00, 11366.68it/s]
[+] Loaded fp16 base: 1 safetensors, 338 tensors
[+] Patched vLLM load_weights — LoRA merge from fp16 cache enabled.
[*] Launching LF-GRPO training loop for 100 steps...
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 7,473 | Num Epochs = 1 | Total steps = 100
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 4 x 1) = 4
 "-____-"     Trainable parameters = 10,551,296 of 1,554,265,600 (0.68% trained)
  0% 0/100 [00:00<?, ?it/s]Unsloth: Will smartly offload gradients to save VRAM!

==================================================
GRADIENT INSULATION VERIFICATION REPORT
==================================================
[+] Total trainable parameters: 56
[*] Sample active parameter gradient norms:
  - base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight: grad_norm = 0.000720
  - base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weight: grad_norm = 0.010567
  - base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight: grad_norm = 0.000589
  - base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weight: grad_norm = 0.015110
  - base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weight: grad_norm = 0.000349
[+] Total frozen parameters: 338
[*] Sample frozen parameter gradient norms:
  - base_model.model.model.embed_tokens.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.bias: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.bias: grad_norm = 0.000000
[+] VERIFICATION SUCCESS: 100% gradient insulation and layout configurations confirmed!
==================================================

Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.
  1% 1/100 [00:39<1:04:57, 39.37s/it][*] vLLM sync #1: 28 LoRA-merged weights pushed (28 A / 28 B detected)
  2% 2/100 [01:11<57:35, 35.26s/it]  [*] vLLM sync #2: 28 LoRA-merged weights pushed (28 A / 28 B detected)
  3% 3/100 [01:38<51:01, 31.56s/it][*] vLLM sync #3: 28 LoRA-merged weights pushed (28 A / 28 B detected)
{'loss': -0.0, 'grad_norm': 0.16058090329170227, 'learning_rate': 6e-06, 'rewards/reward_fn': 1.378172817826271, 'reward': 1.378172817826271, 'reward_std': 0.5186246350407601, 'completion_length': 206.525, 'kl': nan, 'epoch': 0.0}
  5% 5/100 [02:39<49:30, 31.26s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 5)
    Unique invented tags: ['br', 'goal']
    Occurrences: [('br', 2), ('goal', 2)]
==================================================
  7% 7/100 [03:26<41:20, 26.67s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 7)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.11201544106006622, 'learning_rate': 1.3500000000000001e-05, 'rewards/reward_fn': 1.6304870307445527, 'reward': 1.6304870307445527, 'reward_std': 0.5006292570382357, 'completion_length': 200.8875, 'kl': 4.160106182098389e-05, 'epoch': 0.01}
 11% 11/100 [05:37<47:09, 31.80s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 11)
    Unique invented tags: ['0']
    Occurrences: [('0', 1)]
==================================================
 14% 14/100 [06:56<40:18, 28.12s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 14)
    Unique invented tags: ['img']
    Occurrences: [('img', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.11259421706199646, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.6730835914611817, 'reward': 1.6730835914611817, 'reward_std': 0.41996997892856597, 'completion_length': 207.9375, 'kl': 0.00021033883094787598, 'epoch': 0.01}
 17% 17/100 [08:26<41:17, 29.85s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 17)
    Unique invented tags: ['span']
    Occurrences: [('span', 2)]
==================================================
 18% 18/100 [08:48<37:24, 27.37s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 18)
    Unique invented tags: ['br']
    Occurrences: [('br', 2)]
==================================================
 19% 19/100 [09:09<34:23, 25.48s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 19)
    Unique invented tags: ['126']
    Occurrences: [('126', 1)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 19)
    Unique invented tags: ['li', 'ul', 'wednesday']
    Occurrences: [('wednesday', 16), ('ul', 1), ('li', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.14138168096542358, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.557115015387535, 'reward': 1.557115015387535, 'reward_std': 0.4992460647597909, 'completion_length': 195.1625, 'kl': 0.0004363000392913818, 'epoch': 0.01}
 20% 20/100 [09:42<36:54, 27.68s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 20)
    Unique invented tags: ['math']
    Occurrences: [('math', 4)]
==================================================
{'loss': 0.0, 'grad_norm': 0.12444468587636948, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1787630409002303, 'reward': 1.1787630409002303, 'reward_std': 0.4258939363062382, 'completion_length': 216.8625, 'kl': 0.0007765233516693115, 'epoch': 0.01}
 25% 25/100 [12:08<36:59, 29.60s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 25)
    Unique invented tags: ['26']
    Occurrences: [('26', 1)]
==================================================
 27% 27/100 [13:09<36:46, 30.22s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 27)
    Unique invented tags: ['20', 'li', 'ol']
    Occurrences: [('li', 6), ('ol', 2), ('20', 1)]
==================================================
 29% 29/100 [14:01<33:23, 28.21s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 29)
    Unique invented tags: ['br', 'span']
    Occurrences: [('span', 1), ('br', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.15086108446121216, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1106182128190993, 'reward': 1.1106182128190993, 'reward_std': 0.33663463247939945, 'completion_length': 190.3875, 'kl': 0.001253652572631836, 'epoch': 0.02}
 31% 31/100 [15:00<33:39, 29.26s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 31)
    Unique invented tags: ['72']
    Occurrences: [('72', 1)]
==================================================
 32% 32/100 [15:23<31:17, 27.60s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 32)
    Unique invented tags: ['Quantity']
    Occurrences: [('Quantity', 10)]
==================================================
 33% 33/100 [15:43<28:18, 25.35s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 33)
    Unique invented tags: ['10']
    Occurrences: [('10', 1)]
==================================================
 34% 34/100 [16:07<27:11, 24.72s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 34)
    Unique invented tags: ['negative8']
    Occurrences: [('negative8', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1501154601573944, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.222353345155716, 'reward': 1.222353345155716, 'reward_std': 0.3488057188689709, 'completion_length': 171.1625, 'kl': 0.0018939971923828125, 'epoch': 0.02}
 38% 38/100 [18:05<31:08, 30.13s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 38)
    Unique invented tags: ['answer']
    Occurrences: [('answer', 2)]
==================================================
 39% 39/100 [18:37<31:01, 30.51s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 39)
    Unique invented tags: ['14']
    Occurrences: [('14', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.19540148973464966, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 0.990735049545765, 'reward': 0.990735049545765, 'reward_std': 0.3502244021743536, 'completion_length': 212.5375, 'kl': nan, 'epoch': 0.02}
 41% 41/100 [19:33<28:45, 29.25s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 41)
    Unique invented tags: ['q']
    Occurrences: [('q', 4)]
==================================================
 43% 43/100 [20:23<25:53, 27.25s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 43)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.16786158084869385, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.157912626862526, 'reward': 1.157912626862526, 'reward_std': 0.28760677520185707, 'completion_length': 181.3375, 'kl': 0.002592003345489502, 'epoch': 0.02}
 48% 48/100 [22:33<22:29, 25.95s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 48)
    Unique invented tags: ['br']
    Occurrences: [('br', 5)]
==================================================
 49% 49/100 [23:05<23:26, 27.57s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 49)
    Unique invented tags: ['math']
    Occurrences: [('math', 6)]
==================================================
{'loss': -0.0, 'grad_norm': 0.11833062767982483, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0545102521777152, 'reward': 1.0545102521777152, 'reward_std': 0.3818264400586486, 'completion_length': 198.4125, 'kl': 0.003004467487335205, 'epoch': 0.03}
 50% 50/100 [23:30<22:22, 26.85s/it][+] Step 50 → save_step_50.zip uploaded to Drive.
 51% 51/100 [23:57<21:53, 26.80s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 51)
    Unique invented tags: ['b', 'math']
    Occurrences: [('math', 16), ('b', 10)]
==================================================
 53% 53/100 [25:00<23:02, 29.43s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 53)
    Unique invented tags: ['5']
    Occurrences: [('5', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1056666374206543, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0871825784444809, 'reward': 1.0871825784444809, 'reward_std': 0.40419960813596845, 'completion_length': 187.7875, 'kl': 0.004013371467590332, 'epoch': 0.03}
 56% 56/100 [26:18<20:06, 27.41s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 56)
    Unique invented tags: ['329']
    Occurrences: [('329', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.16418574750423431, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 0.9709116131067276, 'reward': 0.9709116131067276, 'reward_std': 0.43575930297374726, 'completion_length': 207.675, 'kl': 0.004972529411315918, 'epoch': 0.03}
 60% 60/100 [28:13<18:35, 27.89s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 60)
    Unique invented tags: ['br']
    Occurrences: [('br', 7)]
==================================================
 61% 61/100 [28:43<18:28, 28.42s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 61)
    Unique invented tags: ['number']
    Occurrences: [('number', 6)]
==================================================
 64% 64/100 [30:05<16:29, 27.49s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 64)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 64)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 4)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1496666520833969, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.024615052342415, 'reward': 1.024615052342415, 'reward_std': 0.483980596344918, 'completion_length': 192.175, 'kl': 0.008082389831542969, 'epoch': 0.03}
 68% 68/100 [31:52<14:38, 27.45s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 68)
    Unique invented tags: ['p']
    Occurrences: [('p', 6)]
==================================================
{'loss': 0.0, 'grad_norm': 0.14877651631832123, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1361784398555757, 'reward': 1.1361784398555757, 'reward_std': 0.38883010791614653, 'completion_length': 191.4, 'kl': 0.009475421905517579, 'epoch': 0.04}
 70% 70/100 [32:48<13:56, 27.88s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 70)
    Unique invented tags: ['math']
    Occurrences: [('math', 3)]
==================================================
{'loss': -0.0, 'grad_norm': 0.12769801914691925, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2117233484983445, 'reward': 1.2117233484983445, 'reward_std': 0.32035922557115554, 'completion_length': 178.0375, 'kl': 0.013519001007080079, 'epoch': 0.04}
 76% 76/100 [35:21<10:54, 27.28s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 76)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
 78% 78/100 [36:12<09:40, 26.39s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 78)
    Unique invented tags: ['6']
    Occurrences: [('6', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1377137452363968, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0506790712475778, 'reward': 1.0506790712475778, 'reward_std': 0.4122882604598999, 'completion_length': 190.7, 'kl': 0.014503288269042968, 'epoch': 0.04}
 82% 82/100 [37:54<07:44, 25.81s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 82)
    Unique invented tags: ['span']
    Occurrences: [('span', 1)]
==================================================
 84% 84/100 [38:55<07:30, 28.18s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 84)
    Unique invented tags: ['br']
    Occurrences: [('br', 4)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 84)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.12473271787166595, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1463520377874374, 'reward': 1.1463520377874374, 'reward_std': 0.3570370342582464, 'completion_length': 204.9125, 'kl': nan, 'epoch': 0.05}
{'loss': -0.0, 'grad_norm': 0.1739737093448639, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1449992835521698, 'reward': 1.1449992835521698, 'reward_std': 0.3882930914871395, 'completion_length': 175.025, 'kl': 0.013361549377441407, 'epoch': 0.05}
 91% 91/100 [42:05<04:02, 26.92s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 91)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 2)]
==================================================
 92% 92/100 [42:34<03:39, 27.39s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 92)
    Unique invented tags: ['li', 'ol']
    Occurrences: [('li', 8), ('ol', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.09428231418132782, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1352285712957382, 'reward': 1.1352285712957382, 'reward_std': 0.35253034736961125, 'completion_length': 212.475, 'kl': 0.01252288818359375, 'epoch': 0.05}
 99% 99/100 [46:04<00:28, 28.87s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 99)
    Unique invented tags: ['30']
    Occurrences: [('30', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1904633492231369, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1314430937170983, 'reward': 1.1314430937170983, 'reward_std': 0.4072277694009244, 'completion_length': 187.175, 'kl': 0.0127838134765625, 'epoch': 0.05}
100% 100/100 [46:25<00:00, 26.28s/it][+] Step 100 → save_step_100.zip uploaded to Drive.
{'train_runtime': 2790.6011, 'train_samples_per_second': 0.143, 'train_steps_per_second': 0.036, 'train_loss': -1.378722174649738e-08, 'epoch': 0.05}
100% 100/100 [46:30<00:00, 27.91s/it]
[*] Saving final LoRA adapters to ./grpo_cot_resumed/final_lora...
[+] LF-GRPO Pipeline Execution Completed Successfully!
[rank0]:[W530 15:35:59.838482472 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
