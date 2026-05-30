[+] Successfully wrote global sitecustomize.py to /usr/local/lib/python3.12/dist-packages/sitecustomize.py
2026-05-30 11:24:08.708198: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO 05-30 11:24:18 __init__.py:190] Automatically detected platform cuda.
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
Generating train split: 7473 examples [00:00, 70961.63 examples/s]
[*] GRPOConfig has no 'vllm_mode' param — skipping.
[*] GRPOConfig has no 'vllm_enforce_eager' param — skipping.
[*] vLLM fp16 inference model: Qwen/Qwen2.5-1.5B-Instruct
[+] Drive already mounted → /content/drive/MyDrive/grpo_checkpoints/grpo_cot_output
[*] vLLM: overriding model 'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit' → 'Qwen/Qwen2.5-1.5B-Instruct'
`torch_dtype` is deprecated! Use `dtype` instead!
WARNING 05-30 11:25:06 config.py:2386] Casting torch.bfloat16 to torch.float16.
INFO 05-30 11:25:23 config.py:542] This model supports multiple tasks: {'reward', 'generate', 'embed', 'classify', 'score'}. Defaulting to 'generate'.
INFO 05-30 11:25:23 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=Qwen/Qwen2.5-1.5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False, 
INFO 05-30 11:25:24 cuda.py:179] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 05-30 11:25:24 cuda.py:227] Using XFormers backend.
INFO 05-30 11:25:25 model_runner.py:1110] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
INFO 05-30 11:25:25 weight_utils.py:252] Using model weights format ['*.safetensors']
INFO 05-30 11:25:25 weight_utils.py:297] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards: 100% 1/1 [00:12<00:00, 12.83s/it]
INFO 05-30 11:25:39 model_runner.py:1115] Loading model weights took 2.8875 GB
INFO 05-30 11:25:45 worker.py:267] Memory profiling takes 5.31 seconds
INFO 05-30 11:25:45 worker.py:267] the current vLLM instance can use total_gpu_memory (14.56GiB) x gpu_memory_utilization (0.50) = 7.28GiB
INFO 05-30 11:25:45 worker.py:267] model weights take 2.89GiB; non_torch_memory takes 0.01GiB; PyTorch activation peak memory takes 2.02GiB; the rest of the memory reserved for KV Cache is 2.36GiB.
INFO 05-30 11:25:45 executor_base.py:110] # CUDA blocks: 5519, # CPU blocks: 9362
INFO 05-30 11:25:45 executor_base.py:115] Maximum concurrency for 32768 tokens per request: 2.69x
INFO 05-30 11:25:51 model_runner.py:1434] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100% 35/35 [00:43<00:00,  1.24s/it]
INFO 05-30 11:26:34 model_runner.py:1562] Graph capturing finished in 44 secs, took 0.17 GiB
INFO 05-30 11:26:34 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 55.72 seconds
Fetching 1 files: 100% 1/1 [00:00<00:00, 11096.04it/s]
[+] Loaded fp16 base: 1 safetensors, 338 tensors
[+] Patched vLLM load_weights — LoRA merge from fp16 cache enabled.
[*] Launching LF-GRPO training loop for 200 steps...
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 7,473 | Num Epochs = 1 | Total steps = 200
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 4 x 1) = 4
 "-____-"     Trainable parameters = 10,551,296 of 1,554,265,600 (0.68% trained)
  0% 0/200 [00:00<?, ?it/s]Unsloth: Will smartly offload gradients to save VRAM!

==================================================
GRADIENT INSULATION VERIFICATION REPORT
==================================================
[+] Total trainable parameters: 56
[*] Sample active parameter gradient norms:
  - base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weight: grad_norm = 0.004315
  - base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weight: grad_norm = 0.006575
  - base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weight: grad_norm = 0.000000
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
  0% 1/200 [00:34<1:52:48, 34.01s/it][*] vLLM sync #1: 28 LoRA-merged weights pushed (28 A / 28 B detected)
  1% 2/200 [01:07<1:51:03, 33.66s/it][*] vLLM sync #2: 28 LoRA-merged weights pushed (28 A / 28 B detected)

==================================================
[!] EMERGENT TAGS DETECTED (Step 2)
    Unique invented tags: ['1']
    Occurrences: [('1', 1)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 2)
    Unique invented tags: ['average_per_day']
    Occurrences: [('average_per_day', 2)]
==================================================
  2% 3/200 [01:40<1:50:05, 33.53s/it][*] vLLM sync #3: 28 LoRA-merged weights pushed (28 A / 28 B detected)
{'loss': 0.0, 'grad_norm': 0.1259138137102127, 'learning_rate': 6e-06, 'rewards/reward_fn': 1.0199999783188105, 'reward': 1.0199999783188105, 'reward_std': 0.780020083207637, 'completion_length': 230.1, 'kl': nan, 'epoch': 0.0}
  3% 6/200 [03:07<1:35:46, 29.62s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 6)
    Unique invented tags: ['1900']
    Occurrences: [('1900', 1)]
==================================================
  4% 9/200 [04:27<1:31:01, 28.59s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 9)
    Unique invented tags: ['21']
    Occurrences: [('21', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.10884794592857361, 'learning_rate': 1.3500000000000001e-05, 'rewards/reward_fn': 1.1760354042053223, 'reward': 1.1760354042053223, 'reward_std': 0.7846149779856205, 'completion_length': 194.4, 'kl': nan, 'epoch': 0.01}
  5% 10/200 [04:49<1:24:08, 26.57s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 10)
    Unique invented tags: ['5']
    Occurrences: [('5', 2)]
==================================================
  6% 13/200 [06:24<1:30:14, 28.95s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 13)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 2)]
==================================================
  7% 14/200 [06:55<1:31:45, 29.60s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 14)
    Unique invented tags: ['56']
    Occurrences: [('56', 1)]
==================================================
{'loss': -0.0, 'grad_norm': nan, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0339811265468597, 'reward': 1.0339811265468597, 'reward_std': 0.8466540291905403, 'completion_length': 227.3, 'kl': nan, 'epoch': 0.01}
  8% 15/200 [07:27<1:34:11, 30.55s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 15)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
  8% 17/200 [08:31<1:35:24, 31.28s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 17)
    Unique invented tags: ['150', '201']
    Occurrences: [('150', 1), ('201', 1)]
==================================================
  9% 18/200 [09:00<1:32:54, 30.63s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 18)
    Unique invented tags: ['half']
    Occurrences: [('half', 1)]
==================================================
 10% 19/200 [09:29<1:31:09, 30.22s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 19)
    Unique invented tags: ['29']
    Occurrences: [('29', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.9483622908592224, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2149937435984612, 'reward': 1.2149937435984612, 'reward_std': 0.6189412884414196, 'completion_length': 215.875, 'kl': 0.005262225866317749, 'epoch': 0.01}
 11% 22/200 [10:41<1:17:36, 26.16s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 22)
    Unique invented tags: ['124']
    Occurrences: [('124', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1112002432346344, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2337512508034707, 'reward': 1.2337512508034707, 'reward_std': 0.5802792310714722, 'completion_length': 224.0625, 'kl': 0.008883348107337952, 'epoch': 0.01}
 13% 26/200 [12:46<1:26:30, 29.83s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 26)
    Unique invented tags: ['4']
    Occurrences: [('4', 1)]
==================================================
 14% 28/200 [13:41<1:21:08, 28.30s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 28)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
 14% 29/200 [14:09<1:20:23, 28.21s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 29)
    Unique invented tags: ['B']
    Occurrences: [('B', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 29)
    Unique invented tags: ['br']
    Occurrences: [('br', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 29)
    Unique invented tags: ['math']
    Occurrences: [('math', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.13857905566692352, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3723749965429306, 'reward': 1.3723749965429306, 'reward_std': 0.5630975671112537, 'completion_length': 205.1, 'kl': 0.007648861408233643, 'epoch': 0.02}
 15% 30/200 [14:36<1:18:49, 27.82s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 30)
    Unique invented tags: ['math']
    Occurrences: [('math', 6)]
==================================================
 16% 31/200 [15:04<1:18:31, 27.88s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 31)
    Unique invented tags: ['math']
    Occurrences: [('math', 2)]
==================================================
 17% 34/200 [16:09<1:04:33, 23.33s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 34)
    Unique invented tags: ['math']
    Occurrences: [('math', 5)]
==================================================
{'loss': 0.0, 'grad_norm': 0.14436915516853333, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.5351679176092148, 'reward': 1.5351679176092148, 'reward_std': 0.4072639432735741, 'completion_length': 175.3625, 'kl': 0.0287283718585968, 'epoch': 0.02}
 18% 36/200 [16:56<1:04:29, 23.59s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 36)
    Unique invented tags: ['br']
    Occurrences: [('br', 3)]
==================================================
 18% 37/200 [17:33<1:14:35, 27.46s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 37)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
 19% 38/200 [18:02<1:15:46, 28.07s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 38)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1309797763824463, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.4188749819993973, 'reward': 1.4188749819993973, 'reward_std': 0.4799067214131355, 'completion_length': 216.975, 'kl': nan, 'epoch': 0.02}
 21% 42/200 [19:36<1:03:19, 24.05s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 42)
    Unique invented tags: ['math']
    Occurrences: [('math', 2)]
==================================================
 22% 43/200 [20:05<1:07:09, 25.67s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 43)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.26627054810523987, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.5781666576862334, 'reward': 1.5781666576862334, 'reward_std': 0.46940327286720274, 'completion_length': 182.2375, 'kl': 0.767970609664917, 'epoch': 0.02}
 24% 49/200 [22:53<1:11:27, 28.40s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 49)
    Unique invented tags: ['72']
    Occurrences: [('72', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1086675301194191, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.4851682603359222, 'reward': 1.4851682603359222, 'reward_std': 0.48738884180784225, 'completion_length': 210.1125, 'kl': 0.02458881139755249, 'epoch': 0.03}
 25% 50/200 [23:19<1:09:10, 27.67s/it][+] Step 50 → save_step_50.zip uploaded to Drive.
 27% 54/200 [25:18<1:09:17, 28.48s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 54)
    Unique invented tags: ['strong']
    Occurrences: [('strong', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.10589171200990677, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.5535416521131993, 'reward': 1.5535416521131993, 'reward_std': 0.4273834068328142, 'completion_length': 207.925, 'kl': 0.004354429244995117, 'epoch': 0.03}
 28% 57/200 [26:43<1:09:17, 29.07s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 57)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3952916711568832, 'reward': 1.3952916711568832, 'reward_std': 0.4938937831670046, 'completion_length': 207.6375, 'kl': nan, 'epoch': 0.03}
 32% 64/200 [29:55<1:02:21, 27.51s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 64)
    Unique invented tags: ['50']
    Occurrences: [('50', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.15520991384983063, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.4955416589975357, 'reward': 1.4955416589975357, 'reward_std': 0.5398214291781187, 'completion_length': 201.2375, 'kl': nan, 'epoch': 0.03}
 34% 68/200 [31:39<58:59, 26.81s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 68)
    Unique invented tags: ['br', 'math']
    Occurrences: [('br', 12), ('math', 8)]
==================================================
{'loss': 0.0, 'grad_norm': 0.13226288557052612, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.695295649766922, 'reward': 1.695295649766922, 'reward_std': 0.43118486665189265, 'completion_length': 205.2625, 'kl': 0.0251187801361084, 'epoch': 0.04}
 36% 73/200 [33:50<52:43, 24.91s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 73)
    Unique invented tags: ['br', 'p']
    Occurrences: [('p', 16), ('br', 4)]
==================================================
 37% 74/200 [34:20<55:10, 26.27s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 74)
    Unique invented tags: ['div']
    Occurrences: [('div', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 74)
    Unique invented tags: ['25']
    Occurrences: [('25', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.11853742599487305, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.6062500029802322, 'reward': 1.6062500029802322, 'reward_std': 0.39426265731453897, 'completion_length': 188.35, 'kl': 0.004353928565979004, 'epoch': 0.04}
 38% 75/200 [34:47<55:00, 26.40s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 75)
    Unique invented tags: ['20']
    Occurrences: [('20', 1)]
==================================================
 38% 76/200 [35:18<57:36, 27.87s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 76)
    Unique invented tags: ['300']
    Occurrences: [('300', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.13639946281909943, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.4994179099798202, 'reward': 1.4994179099798202, 'reward_std': 0.4401446122676134, 'completion_length': 205.15, 'kl': 0.005084514617919922, 'epoch': 0.04}
 40% 81/200 [37:28<50:25, 25.43s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 81)
    Unique invented tags: ['32']
    Occurrences: [('32', 1)]
==================================================
 42% 84/200 [38:53<51:59, 26.89s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 84)
    Unique invented tags: ['84', 'table', 'tbody', 'td', 'th', 'thead', 'tr']
    Occurrences: [('td', 30), ('th', 10), ('tr', 8), ('84', 2), ('table', 2), ('thead', 2), ('tbody', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.09977070242166519, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.5782574445009232, 'reward': 1.5782574445009232, 'reward_std': 0.4279792245477438, 'completion_length': 216.8375, 'kl': 0.019080066680908205, 'epoch': 0.05}
 44% 87/200 [40:15<51:25, 27.31s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 87)
    Unique invented tags: ['45', '537']
    Occurrences: [('537', 1), ('45', 1)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 87)
    Unique invented tags: ['number']
    Occurrences: [('number', 4)]
==================================================
 44% 89/200 [41:13<52:17, 28.27s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 89)
    Unique invented tags: ['b']
    Occurrences: [('b', 4)]
==================================================
{'loss': -0.0, 'grad_norm': 0.11341039836406708, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.621374985575676, 'reward': 1.621374985575676, 'reward_std': 0.4377073831856251, 'completion_length': 202.175, 'kl': nan, 'epoch': 0.05}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.5752512246370316, 'reward': 1.5752512246370316, 'reward_std': 0.523149936273694, 'completion_length': 221.45, 'kl': nan, 'epoch': 0.05}
{'loss': -0.0, 'grad_norm': 0.09973075240850449, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.6550416320562362, 'reward': 1.6550416320562362, 'reward_std': 0.5116925850510597, 'completion_length': 194.725, 'kl': 0.02657477855682373, 'epoch': 0.05}
 50% 100/200 [46:10<43:06, 25.86s/it][+] Step 100 → save_step_100.zip uploaded to Drive.
 52% 103/200 [47:35<43:45, 27.06s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 103)
    Unique invented tags: ['math']
    Occurrences: [('math', 25)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1450546681880951, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.6691249579191207, 'reward': 1.6691249579191207, 'reward_std': 0.4593825988471508, 'completion_length': 196.5375, 'kl': 0.011848354339599609, 'epoch': 0.06}
 52% 105/200 [48:26<41:30, 26.21s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 105)
    Unique invented tags: ['36']
    Occurrences: [('36', 1)]
==================================================
 54% 107/200 [49:16<39:55, 25.76s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 107)
    Unique invented tags: ['br']
    Occurrences: [('br', 5)]
==================================================
 54% 108/200 [49:45<41:00, 26.75s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 108)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
 55% 109/200 [50:09<39:13, 25.87s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 109)
    Unique invented tags: ['1', '2', '3', '4', '5']
    Occurrences: [('1', 2), ('2', 1), ('3', 1), ('4', 1), ('5', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.11290297657251358, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.5172916457057, 'reward': 1.5172916457057, 'reward_std': 0.5135372739285231, 'completion_length': 207.075, 'kl': nan, 'epoch': 0.06}
{'loss': -0.0, 'grad_norm': 0.132577583193779, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.6231249749660492, 'reward': 1.6231249749660492, 'reward_std': 0.5434322714805603, 'completion_length': 190.9, 'kl': 0.020946979522705078, 'epoch': 0.06}
 60% 119/200 [54:17<32:21, 23.98s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 119)
    Unique invented tags: ['tool_call']
    Occurrences: [('tool_call', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.36448317766189575, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.7619583249092101, 'reward': 1.7619583249092101, 'reward_std': 0.4892299797385931, 'completion_length': 188.35, 'kl': nan, 'epoch': 0.06}
 61% 122/200 [55:29<30:19, 23.33s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 122)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.11885733157396317, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3381249919533729, 'reward': 1.3381249919533729, 'reward_std': 0.30650622490793467, 'completion_length': 176.875, 'kl': 0.11801471710205078, 'epoch': 0.07}
{'loss': 0.0, 'grad_norm': 0.1404573917388916, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0937916740775109, 'reward': 1.0937916740775109, 'reward_std': 0.36886969208717346, 'completion_length': 221.7125, 'kl': nan, 'epoch': 0.07}
 65% 130/200 [59:05<30:43, 26.34s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 130)
    Unique invented tags: ['code', 'li', 'ol', 'space']
    Occurrences: [('space', 8), ('li', 6), ('ol', 2), ('code', 2)]
==================================================
 66% 131/200 [59:31<29:58, 26.06s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 131)
    Unique invented tags: ['33']
    Occurrences: [('33', 1)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 131)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
 66% 132/200 [59:55<28:51, 25.46s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 132)
    Unique invented tags: ['math']
    Occurrences: [('math', 4)]
==================================================
 66% 133/200 [1:00:17<27:24, 24.54s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 133)
    Unique invented tags: ['details', 'summary']
    Occurrences: [('details', 6), ('summary', 6)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 133)
    Unique invented tags: ['math']
    Occurrences: [('math', 8)]
==================================================
 67% 134/200 [1:00:47<28:37, 26.03s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 134)
    Unique invented tags: ['math']
    Occurrences: [('math', 10)]
==================================================
{'loss': 0.0, 'grad_norm': 0.12709571421146393, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1917083591222764, 'reward': 1.1917083591222764, 'reward_std': 0.36024548253044486, 'completion_length': 185.05, 'kl': 0.06303796768188477, 'epoch': 0.07}
 69% 138/200 [1:02:17<25:13, 24.41s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 138)
    Unique invented tags: ['center']
    Occurrences: [('center', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 138)
    Unique invented tags: ['br']
    Occurrences: [('br', 12)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1485108733177185, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3183750092983246, 'reward': 1.3183750092983246, 'reward_std': 0.35175214512273667, 'completion_length': 183.225, 'kl': 0.04428253173828125, 'epoch': 0.07}
 70% 140/200 [1:03:12<25:54, 25.91s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 140)
    Unique invented tags: ['div']
    Occurrences: [('div', 3)]
==================================================
 70% 141/200 [1:03:34<24:10, 24.59s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 141)
    Unique invented tags: ['br']
    Occurrences: [('br', 1)]
==================================================
 71% 142/200 [1:04:03<25:01, 25.88s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 142)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.16282148659229279, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2822499930858613, 'reward': 1.2822499930858613, 'reward_std': 0.2942908713594079, 'completion_length': 185.7375, 'kl': 0.055138587951660156, 'epoch': 0.08}
 72% 145/200 [1:05:10<21:53, 23.87s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 145)
    Unique invented tags: ['5']
    Occurrences: [('5', 1)]
==================================================
 73% 146/200 [1:05:33<21:14, 23.59s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 146)
    Unique invented tags: ['li', 'ol', 'ul']
    Occurrences: [('li', 20), ('ol', 8), ('ul', 6)]
==================================================
 74% 148/200 [1:06:23<20:58, 24.20s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 148)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.17794686555862427, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3036250084638596, 'reward': 1.3036250084638596, 'reward_std': 0.35837058164179325, 'completion_length': 180.0625, 'kl': 0.06744003295898438, 'epoch': 0.08}
 75% 150/200 [1:07:17<21:27, 25.74s/it][+] Step 150 → save_step_150.zip uploaded to Drive.
 77% 154/200 [1:09:01<19:45, 25.77s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 154)
    Unique invented tags: ['answer']
    Occurrences: [('answer', 12)]
==================================================
{'loss': 0.0, 'grad_norm': 0.11868132650852203, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0923333495855332, 'reward': 1.0923333495855332, 'reward_std': 0.30480487337335943, 'completion_length': 201.1375, 'kl': 0.08179130554199218, 'epoch': 0.08}
 78% 157/200 [1:10:29<19:34, 27.32s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 157)
    Unique invented tags: ['30', 'br']
    Occurrences: [('br', 2), ('30', 1)]
==================================================
 79% 158/200 [1:10:59<19:41, 28.12s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 158)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
 80% 159/200 [1:11:25<18:41, 27.36s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 159)
    Unique invented tags: ['br']
    Occurrences: [('br', 9)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1834438294172287, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1198333472013473, 'reward': 1.1198333472013473, 'reward_std': 0.4150658829137683, 'completion_length': 190.5, 'kl': 0.0870626449584961, 'epoch': 0.09}
 80% 160/200 [1:11:49<17:36, 26.42s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 160)
    Unique invented tags: ['span']
    Occurrences: [('span', 18)]
==================================================
{'loss': -0.0, 'grad_norm': 0.12028966844081879, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2155432254076004, 'reward': 1.2155432254076004, 'reward_std': 0.30932968044653536, 'completion_length': 196.4875, 'kl': 0.11556797027587891, 'epoch': 0.09}
 84% 168/200 [1:15:23<13:55, 26.10s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 168)
    Unique invented tags: ['2175']
    Occurrences: [('2175', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.12593410909175873, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2084583356976508, 'reward': 1.2084583356976508, 'reward_std': 0.2681766549125314, 'completion_length': 206.45, 'kl': 0.07145500183105469, 'epoch': 0.09}
 86% 173/200 [1:17:38<11:41, 25.98s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 173)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.16084928810596466, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.173833355307579, 'reward': 1.173833355307579, 'reward_std': 0.3755111857317388, 'completion_length': 171.0875, 'kl': 0.055147552490234376, 'epoch': 0.09}
 88% 177/200 [1:19:04<08:44, 22.81s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 177)
    Unique invented tags: ['math']
    Occurrences: [('math', 2)]
==================================================
 89% 178/200 [1:19:31<08:53, 24.24s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 178)
    Unique invented tags: ['5']
    Occurrences: [('5', 1)]
==================================================
 90% 179/200 [1:19:50<07:52, 22.51s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 179)
    Unique invented tags: ['span']
    Occurrences: [('span', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.14722201228141785, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.269749990105629, 'reward': 1.269749990105629, 'reward_std': 0.25550043387338517, 'completion_length': 172.1875, 'kl': 0.11760902404785156, 'epoch': 0.1}
 91% 182/200 [1:21:09<07:31, 25.10s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 182)
    Unique invented tags: ['129']
    Occurrences: [('129', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.2219616025686264, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0302083477377892, 'reward': 1.0302083477377892, 'reward_std': 0.31541233602911234, 'completion_length': 205.1, 'kl': 0.10538330078125, 'epoch': 0.1}
 94% 188/200 [1:23:55<05:20, 26.75s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 188)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1364213526248932, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.11405159085989, 'reward': 1.11405159085989, 'reward_std': 0.39670213172212243, 'completion_length': 225.0625, 'kl': 0.5448413848876953, 'epoch': 0.1}
 95% 190/200 [1:25:01<05:02, 30.25s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 190)
    Unique invented tags: ['final_answer']
    Occurrences: [('final_answer', 2)]
==================================================
 96% 191/200 [1:25:29<04:25, 29.52s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 191)
    Unique invented tags: ['br']
    Occurrences: [('br', 2)]
==================================================
 96% 192/200 [1:26:03<04:05, 30.70s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 192)
    Unique invented tags: ['br']
    Occurrences: [('br', 1)]
==================================================
 96% 193/200 [1:26:30<03:28, 29.73s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 193)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 28)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 193)
    Unique invented tags: ['good']
    Occurrences: [('good', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.24156512320041656, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2398762673139572, 'reward': 1.2398762673139572, 'reward_std': 0.38103729449212553, 'completion_length': 220.0375, 'kl': 0.7832839965820313, 'epoch': 0.1}
 98% 195/200 [1:27:24<02:19, 27.95s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 195)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 195)
    Unique invented tags: ['hr', 'number']
    Occurrences: [('number', 2), ('hr', 1)]
==================================================
 98% 196/200 [1:27:48<01:47, 26.79s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 196)
    Unique invented tags: ['math', 'number']
    Occurrences: [('math', 41), ('number', 16)]
==================================================
 98% 197/200 [1:28:19<01:23, 27.96s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 197)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 197)
    Unique invented tags: ['plain']
    Occurrences: [('plain', 45)]
==================================================
100% 199/200 [1:29:14<00:27, 27.54s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 199)
    Unique invented tags: ['760']
    Occurrences: [('760', 1)]
==================================================
{'loss': -0.0, 'grad_norm': nan, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2750024884939193, 'reward': 1.2750024884939193, 'reward_std': 0.3208400834351778, 'completion_length': 189.3375, 'kl': nan, 'epoch': 0.11}
100% 200/200 [1:29:39<00:00, 26.72s/it][+] Step 200 → save_step_200.zip uploaded to Drive.
{'train_runtime': 5383.6559, 'train_samples_per_second': 0.149, 'train_steps_per_second': 0.037, 'train_loss': -1.4708066853330592e-08, 'epoch': 0.11}
100% 200/200 [1:29:43<00:00, 26.92s/it]
[*] Saving final LoRA adapters to ./grpo_cot_output/final_lora...
[+] LF-GRPO Pipeline Execution Completed Successfully!
[rank0]:[W530 12:56:26.175483491 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
