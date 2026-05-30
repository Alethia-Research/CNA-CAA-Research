[+] Successfully wrote global sitecustomize.py to /usr/local/lib/python3.12/dist-packages/sitecustomize.py
2026-05-30 16:21:09.761528: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO 05-30 16:21:15 __init__.py:190] Automatically detected platform cuda.
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
[*] Loading adapter weights from /content/grpo_cot_resumed_final/final_lora...
[+] Adapter weights loaded — stage-1 weights restored, continuing from step 100.
Generating train split: 7473 examples [00:00, 143315.18 examples/s]
[*] GRPOConfig has no 'vllm_mode' param — skipping.
[*] GRPOConfig has no 'vllm_enforce_eager' param — skipping.
[*] vLLM fp16 inference model: Qwen/Qwen2.5-1.5B-Instruct
[+] Drive already mounted → /content/drive/MyDrive/grpo_checkpoints/grpo_cot_resumed_final
[*] vLLM: overriding model 'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit' → 'Qwen/Qwen2.5-1.5B-Instruct'
config.json: 100% 660/660 [00:00<00:00, 4.63MB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
WARNING 05-30 16:22:03 config.py:2386] Casting torch.bfloat16 to torch.float16.
INFO 05-30 16:22:20 config.py:542] This model supports multiple tasks: {'embed', 'score', 'generate', 'reward', 'classify'}. Defaulting to 'generate'.
INFO 05-30 16:22:20 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=Qwen/Qwen2.5-1.5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False, 
tokenizer_config.json: 7.30kB [00:00, 34.5MB/s]
vocab.json: 2.78MB [00:00, 93.5MB/s]
merges.txt: 1.67MB [00:00, 149MB/s]
tokenizer.json: 7.03MB [00:00, 179MB/s]
generation_config.json: 100% 242/242 [00:00<00:00, 1.40MB/s]
INFO 05-30 16:22:23 cuda.py:179] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 05-30 16:22:23 cuda.py:227] Using XFormers backend.
INFO 05-30 16:22:24 model_runner.py:1110] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
INFO 05-30 16:22:25 weight_utils.py:252] Using model weights format ['*.safetensors']
model.safetensors: 100% 3.09G/3.09G [00:43<00:00, 70.9MB/s]
INFO 05-30 16:23:09 weight_utils.py:297] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards: 100% 1/1 [00:11<00:00, 11.97s/it]
INFO 05-30 16:23:22 model_runner.py:1115] Loading model weights took 2.8875 GB
INFO 05-30 16:23:28 worker.py:267] Memory profiling takes 5.07 seconds
INFO 05-30 16:23:28 worker.py:267] the current vLLM instance can use total_gpu_memory (14.56GiB) x gpu_memory_utilization (0.50) = 7.28GiB
INFO 05-30 16:23:28 worker.py:267] model weights take 2.89GiB; non_torch_memory takes 0.01GiB; PyTorch activation peak memory takes 2.02GiB; the rest of the memory reserved for KV Cache is 2.36GiB.
INFO 05-30 16:23:28 executor_base.py:110] # CUDA blocks: 5519, # CPU blocks: 9362
INFO 05-30 16:23:28 executor_base.py:115] Maximum concurrency for 32768 tokens per request: 2.69x
INFO 05-30 16:23:35 model_runner.py:1434] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100% 35/35 [00:42<00:00,  1.23s/it]
INFO 05-30 16:24:18 model_runner.py:1562] Graph capturing finished in 43 secs, took 0.17 GiB
INFO 05-30 16:24:18 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 56.26 seconds
Fetching 1 files: 100% 1/1 [00:00<00:00, 591.75it/s]
[+] Loaded fp16 base: 1 safetensors, 338 tensors
[+] Patched vLLM load_weights — LoRA merge from fp16 cache enabled.
[*] Launching LF-GRPO training loop for 300 steps...
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 7,473 | Num Epochs = 1 | Total steps = 300
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 4 x 1) = 4
 "-____-"     Trainable parameters = 10,551,296 of 1,554,265,600 (0.68% trained)
  0% 0/300 [00:00<?, ?it/s]Unsloth: Will smartly offload gradients to save VRAM!

==================================================
GRADIENT INSULATION VERIFICATION REPORT
==================================================
[+] Total trainable parameters: 56
[*] Sample active parameter gradient norms:
  - base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight: grad_norm = 0.001058
  - base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weight: grad_norm = 0.011731
  - base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight: grad_norm = 0.001052
  - base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weight: grad_norm = 0.015346
  - base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weight: grad_norm = 0.000563
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
  0% 1/300 [00:40<3:23:45, 40.89s/it][*] vLLM sync #1: 28 LoRA-merged weights pushed (28 A / 28 B detected)
  1% 2/300 [01:08<2:44:09, 33.05s/it][*] vLLM sync #2: 28 LoRA-merged weights pushed (28 A / 28 B detected)

==================================================
[!] EMERGENT TAGS DETECTED (Step 2)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
  1% 3/300 [01:33<2:24:37, 29.22s/it][*] vLLM sync #3: 28 LoRA-merged weights pushed (28 A / 28 B detected)

==================================================
[!] EMERGENT TAGS DETECTED (Step 3)
    Unique invented tags: ['80']
    Occurrences: [('80', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.15726783871650696, 'learning_rate': 6e-06, 'rewards/reward_fn': 1.5948310494422913, 'reward': 1.5948310494422913, 'reward_std': 0.41733839493244884, 'completion_length': 188.0625, 'kl': nan, 'epoch': 0.0}
  3% 9/300 [03:58<2:06:28, 26.08s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 9)
    Unique invented tags: ['div', 'li', 'ol', 'strong']
    Occurrences: [('li', 8), ('div', 2), ('strong', 2), ('ol', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.13281530141830444, 'learning_rate': 1.3500000000000001e-05, 'rewards/reward_fn': 1.7179407238960267, 'reward': 1.7179407238960267, 'reward_std': 0.4982848659157753, 'completion_length': 193.1875, 'kl': 3.7163496017456055e-05, 'epoch': 0.01}
  3% 10/300 [04:27<2:10:36, 27.02s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 10)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
  4% 11/300 [05:00<2:19:27, 28.95s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 11)
    Unique invented tags: ['span']
    Occurrences: [('span', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.13187433779239655, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.7271795094013214, 'reward': 1.7271795094013214, 'reward_std': 0.45520185939967633, 'completion_length': 189.9, 'kl': 0.00024048686027526854, 'epoch': 0.01}
{'loss': -0.0, 'grad_norm': 0.14204853773117065, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.9155282080173492, 'reward': 1.9155282080173492, 'reward_std': 0.354338906519115, 'completion_length': 175.2625, 'kl': 0.0003554433584213257, 'epoch': 0.01}
{'loss': -0.0, 'grad_norm': 0.1330159306526184, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2515117280185222, 'reward': 1.2515117280185222, 'reward_std': 0.3840794278308749, 'completion_length': 199.125, 'kl': 0.0004480898380279541, 'epoch': 0.01}
  9% 27/300 [11:31<1:59:06, 26.18s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 27)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
 10% 29/300 [12:21<1:56:12, 25.73s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 29)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.13572905957698822, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0227152973413467, 'reward': 1.0227152973413467, 'reward_std': 0.3438741970807314, 'completion_length': 184.3125, 'kl': 0.000932765007019043, 'epoch': 0.02}
{'loss': 0.0, 'grad_norm': 0.15957365930080414, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2500793009996414, 'reward': 1.2500793009996414, 'reward_std': 0.29876091398764404, 'completion_length': 162.3875, 'kl': 0.001450490951538086, 'epoch': 0.02}
 12% 37/300 [15:23<1:41:38, 23.19s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 37)
    Unique invented tags: ['span']
    Occurrences: [('span', 202)]
==================================================
{'loss': 0.0, 'grad_norm': 0.20356491208076477, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.05616687387228, 'reward': 1.05616687387228, 'reward_std': 0.4188710363581777, 'completion_length': 190.25, 'kl': 0.0021978378295898437, 'epoch': 0.02}
 14% 43/300 [17:43<1:35:21, 22.26s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 43)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.167120099067688, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2034442394971847, 'reward': 1.2034442394971847, 'reward_std': 0.2705764865502715, 'completion_length': 151.9125, 'kl': 0.002315187454223633, 'epoch': 0.02}
 16% 47/300 [19:09<1:32:39, 21.97s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 47)
    Unique invented tags: ['li', 'ol']
    Occurrences: [('li', 12), ('ol', 8)]
==================================================
{'loss': -0.0, 'grad_norm': 0.10645003616809845, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1119090184569358, 'reward': 1.1119090184569358, 'reward_std': 0.3297112428583205, 'completion_length': 183.0875, 'kl': 0.001786184310913086, 'epoch': 0.03}
 18% 53/300 [21:35<1:40:18, 24.37s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 53)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.13408000767230988, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1540947526693344, 'reward': 1.1540947526693344, 'reward_std': 0.42410874031484125, 'completion_length': 184.275, 'kl': 0.0021122455596923827, 'epoch': 0.03}
{'loss': -0.0, 'grad_norm': 0.15099357068538666, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1478895232081414, 'reward': 1.1478895232081414, 'reward_std': 0.36467730859294534, 'completion_length': 183.1875, 'kl': 0.0031340360641479493, 'epoch': 0.03}
{'loss': -0.0, 'grad_norm': 0.14019332826137543, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.124651950597763, 'reward': 1.124651950597763, 'reward_std': 0.41812853757292034, 'completion_length': 187.2125, 'kl': nan, 'epoch': 0.03}
 22% 67/300 [27:08<1:30:36, 23.33s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 67)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.17196515202522278, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2613537698984145, 'reward': 1.2613537698984145, 'reward_std': 0.32738864328712225, 'completion_length': 179.55, 'kl': nan, 'epoch': 0.04}
{'loss': -0.0, 'grad_norm': 0.12443745136260986, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.212884809076786, 'reward': 1.212884809076786, 'reward_std': 0.36803745599463583, 'completion_length': 170.3625, 'kl': nan, 'epoch': 0.04}
{'loss': -0.0, 'grad_norm': 0.1455964297056198, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1144394800066948, 'reward': 1.1144394800066948, 'reward_std': 0.40487091848626733, 'completion_length': 197.5875, 'kl': nan, 'epoch': 0.04}
{'loss': 0.0, 'grad_norm': 0.1583775132894516, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1866758733987808, 'reward': 1.1866758733987808, 'reward_std': 0.31807997946161776, 'completion_length': 192.5375, 'kl': nan, 'epoch': 0.05}
{'loss': -0.0, 'grad_norm': 0.15280786156654358, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2168425738811492, 'reward': 1.2168425738811492, 'reward_std': 0.2259039467200637, 'completion_length': 181.25, 'kl': 0.0069558143615722655, 'epoch': 0.05}
 31% 92/300 [37:37<1:37:46, 28.21s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 92)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
 31% 93/300 [38:04<1:35:13, 27.60s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 93)
    Unique invented tags: ['sub']
    Occurrences: [('sub', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 93)
    Unique invented tags: ['br']
    Occurrences: [('br', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.12435489147901535, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0404068484902382, 'reward': 1.0404068484902382, 'reward_std': 0.3940002920106053, 'completion_length': 223.4, 'kl': 0.007370376586914062, 'epoch': 0.05}
{'loss': 0.0, 'grad_norm': 0.189840629696846, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.304063293337822, 'reward': 1.304063293337822, 'reward_std': 0.27765539921820165, 'completion_length': 158.825, 'kl': 0.004181957244873047, 'epoch': 0.05}
 33% 100/300 [40:51<1:14:48, 22.44s/it][+] Step 100 → save_step_100.zip uploaded to Drive.
 34% 103/300 [42:13<1:22:48, 25.22s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 103)
    Unique invented tags: ['br']
    Occurrences: [('br', 3)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1666930466890335, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1349183529615403, 'reward': 1.1349183529615403, 'reward_std': 0.3648401769809425, 'completion_length': 174.675, 'kl': 0.004449462890625, 'epoch': 0.06}
{'loss': 0.0, 'grad_norm': 0.14460450410842896, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1261551320552825, 'reward': 1.1261551320552825, 'reward_std': 0.2799194843508303, 'completion_length': 188.7125, 'kl': nan, 'epoch': 0.06}
 38% 114/300 [46:36<1:15:36, 24.39s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 114)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1756776124238968, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.148021411895752, 'reward': 1.148021411895752, 'reward_std': 0.40029212152585386, 'completion_length': 178.4375, 'kl': 0.006151533126831055, 'epoch': 0.06}
 39% 118/300 [48:03<1:07:20, 22.20s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 118)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.12278176844120026, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1547818511724472, 'reward': 1.1547818511724472, 'reward_std': 0.3345178152434528, 'completion_length': 188.3625, 'kl': nan, 'epoch': 0.06}
{'loss': -0.0, 'grad_norm': 0.1364343762397766, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3183403730392456, 'reward': 1.3183403730392456, 'reward_std': 0.30467590428888797, 'completion_length': 176.0125, 'kl': 0.017615604400634765, 'epoch': 0.07}
 42% 127/300 [51:46<1:12:32, 25.16s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 127)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.19437067210674286, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0606279119849205, 'reward': 1.0606279119849205, 'reward_std': 0.33570294380187987, 'completion_length': 207.0, 'kl': nan, 'epoch': 0.07}
{'loss': 0.0, 'grad_norm': 0.15036937594413757, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2282128304243087, 'reward': 1.2282128304243087, 'reward_std': 0.3671085602603853, 'completion_length': 164.35, 'kl': 0.011809444427490235, 'epoch': 0.07}
 45% 135/300 [54:58<59:08, 21.50s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 135)
    Unique invented tags: ['thought']
    Occurrences: [('thought', 1)]
==================================================
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3756836831569672, 'reward': 1.3756836831569672, 'reward_std': 0.29689839500933884, 'completion_length': 177.4625, 'kl': nan, 'epoch': 0.07}
 47% 142/300 [57:40<1:01:53, 23.50s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 142)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.1449396312236786, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.338073444366455, 'reward': 1.338073444366455, 'reward_std': 0.28381677996367216, 'completion_length': 170.2625, 'kl': nan, 'epoch': 0.08}
 49% 147/300 [59:30<57:16, 22.46s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 147)
    Unique invented tags: ['thinking']
    Occurrences: [('thinking', 1)]
==================================================
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3050164945423604, 'reward': 1.3050164945423604, 'reward_std': 0.27492209021002056, 'completion_length': 180.1625, 'kl': nan, 'epoch': 0.08}
 51% 153/300 [1:01:53<54:53, 22.40s/it]  
==================================================
[!] EMERGENT TAGS DETECTED (Step 153)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 153)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.150785431265831, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1915547579526902, 'reward': 1.1915547579526902, 'reward_std': 0.31107675479725005, 'completion_length': 179.0875, 'kl': 0.05274982452392578, 'epoch': 0.08}
{'loss': 0.0, 'grad_norm': 0.1911262571811676, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.180574369430542, 'reward': 1.180574369430542, 'reward_std': 0.32082450883463026, 'completion_length': 188.85, 'kl': 0.01868915557861328, 'epoch': 0.09}
 53% 160/300 [1:04:48<54:56, 23.55s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 160)
    Unique invented tags: ['ans']
    Occurrences: [('ans', 2)]
==================================================
 54% 162/300 [1:05:51<1:03:44, 27.71s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 162)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.16821812093257904, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.2151302814483642, 'reward': 1.2151302814483642, 'reward_std': 0.3258931225165725, 'completion_length': 192.2875, 'kl': nan, 'epoch': 0.09}
{'loss': -0.0, 'grad_norm': 0.15027236938476562, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1457430630922318, 'reward': 1.1457430630922318, 'reward_std': 0.3393503650091588, 'completion_length': 203.95, 'kl': 0.026686477661132812, 'epoch': 0.09}
 57% 172/300 [1:10:06<55:45, 26.14s/it]  
==================================================
[!] EMERGENT TAGS DETECTED (Step 172)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.15419012308120728, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1222255110740662, 'reward': 1.1222255110740662, 'reward_std': 0.43415398290380836, 'completion_length': 192.825, 'kl': 0.08364105224609375, 'epoch': 0.09}
 58% 175/300 [1:11:15<49:53, 23.94s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 175)
    Unique invented tags: ['span']
    Occurrences: [('span', 1)]
==================================================
 59% 178/300 [1:12:23<47:01, 23.13s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 178)
    Unique invented tags: ['br']
    Occurrences: [('br', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.13236446678638458, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1667112678289413, 'reward': 1.1667112678289413, 'reward_std': 0.2643367585260421, 'completion_length': 178.15, 'kl': 0.08180294036865235, 'epoch': 0.1}
{'loss': -0.0, 'grad_norm': 0.20740529894828796, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.0899378061294556, 'reward': 1.0899378061294556, 'reward_std': 0.2995959520339966, 'completion_length': 186.5875, 'kl': 0.2941127777099609, 'epoch': 0.1}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.159583579003811, 'reward': 1.159583579003811, 'reward_std': 0.32895185113884506, 'completion_length': 214.7125, 'kl': nan, 'epoch': 0.1}
{'loss': -0.0, 'grad_norm': 0.18344111740589142, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.161453738808632, 'reward': 1.161453738808632, 'reward_std': 0.29502933118492364, 'completion_length': 196.0125, 'kl': 0.0963815689086914, 'epoch': 0.1}
{'loss': -0.0, 'grad_norm': 0.18256376683712006, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.3114382594823837, 'reward': 1.3114382594823837, 'reward_std': 0.28566997991874815, 'completion_length': 158.45, 'kl': 0.030595779418945312, 'epoch': 0.11}
 67% 200/300 [1:21:26<37:30, 22.50s/it][+] Step 200 → save_step_200.zip uploaded to Drive.
 67% 202/300 [1:22:31<43:16, 26.49s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 202)
    Unique invented tags: ['br']
    Occurrences: [('br', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.2034682035446167, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.149840322136879, 'reward': 1.149840322136879, 'reward_std': 0.34919661274179814, 'completion_length': 184.2, 'kl': 0.02460784912109375, 'epoch': 0.11}
{'loss': -0.0, 'grad_norm': 0.16186697781085968, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.28073807656765, 'reward': 1.28073807656765, 'reward_std': 0.3411110781133175, 'completion_length': 171.0, 'kl': 0.03215599060058594, 'epoch': 0.11}
 71% 214/300 [1:27:23<38:44, 27.03s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 214)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1817377656698227, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1206269532442092, 'reward': 1.1206269532442092, 'reward_std': 0.3883979795500636, 'completion_length': 191.5625, 'kl': 0.0373748779296875, 'epoch': 0.12}
{'loss': 0.0, 'grad_norm': 0.1320083886384964, 'learning_rate': 1.5e-05, 'rewards/reward_fn': 1.1301309168338776, 'reward': 1.1301309168338776, 'reward_std': 0.4201866292860359, 'completion_length': 193.4125, 'kl': 0.035988235473632814, 'epoch': 0.12}
{'loss': -0.0, 'grad_norm': 0.14925488829612732, 'learning_rate': 1.4249999999999999e-05, 'rewards/reward_fn': 1.1218569204211235, 'reward': 1.1218569204211235, 'reward_std': 0.3242927827872336, 'completion_length': 187.1875, 'kl': 0.05489540100097656, 'epoch': 0.12}
 76% 227/300 [1:32:54<32:04, 26.37s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 227)
    Unique invented tags: ['math']
    Occurrences: [('math', 6)]
==================================================
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 1.33125e-05, 'rewards/reward_fn': 1.1002120770514012, 'reward': 1.1002120770514012, 'reward_std': 0.36375359939411284, 'completion_length': 192.2, 'kl': nan, 'epoch': 0.12}
{'loss': -0.0, 'grad_norm': 0.15726716816425323, 'learning_rate': 1.2375e-05, 'rewards/reward_fn': 1.243432766199112, 'reward': 1.243432766199112, 'reward_std': 0.3710964012891054, 'completion_length': 209.0625, 'kl': 0.06396446228027344, 'epoch': 0.13}
{'loss': -0.0, 'grad_norm': 0.18400810658931732, 'learning_rate': 1.14375e-05, 'rewards/reward_fn': 1.1644359886646272, 'reward': 1.1644359886646272, 'reward_std': 0.3260802686214447, 'completion_length': 172.05, 'kl': 0.04627532958984375, 'epoch': 0.13}
{'loss': -0.0, 'grad_norm': 0.13400453329086304, 'learning_rate': 1.05e-05, 'rewards/reward_fn': 1.1834025979042053, 'reward': 1.1834025979042053, 'reward_std': 0.3064890908426605, 'completion_length': 205.6125, 'kl': 0.035906982421875, 'epoch': 0.13}
{'loss': 0.0, 'grad_norm': 0.14081494510173798, 'learning_rate': 9.5625e-06, 'rewards/reward_fn': 1.2185106456279755, 'reward': 1.2185106456279755, 'reward_std': 0.32638459562440403, 'completion_length': 197.75, 'kl': 0.0541046142578125, 'epoch': 0.13}
{'loss': 0.0, 'grad_norm': 0.15956827998161316, 'learning_rate': 8.625e-06, 'rewards/reward_fn': 1.2183078140020371, 'reward': 1.2183078140020371, 'reward_std': 0.31360337720252573, 'completion_length': 186.95, 'kl': nan, 'epoch': 0.14}
{'loss': -0.0, 'grad_norm': 0.11464650928974152, 'learning_rate': 7.687499999999999e-06, 'rewards/reward_fn': 1.1811096325516701, 'reward': 1.1811096325516701, 'reward_std': 0.27747585335746405, 'completion_length': 192.4625, 'kl': 0.07894058227539062, 'epoch': 0.14}
{'loss': 0.0, 'grad_norm': 0.2676941454410553, 'learning_rate': 6.75e-06, 'rewards/reward_fn': 1.3736521691083907, 'reward': 1.3736521691083907, 'reward_std': 0.2587151825428009, 'completion_length': 163.925, 'kl': 0.20014495849609376, 'epoch': 0.14}
{'loss': 0.0, 'grad_norm': 0.20515134930610657, 'learning_rate': 5.8125e-06, 'rewards/reward_fn': 1.096025961637497, 'reward': 1.096025961637497, 'reward_std': 0.41754114544019105, 'completion_length': 211.3, 'kl': 0.4643463134765625, 'epoch': 0.14}
 91% 273/300 [1:52:41<11:46, 26.17s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 273)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.2366938591003418, 'learning_rate': 4.874999999999999e-06, 'rewards/reward_fn': 1.0045414090156555, 'reward': 1.0045414090156555, 'reward_std': 0.26495879059657457, 'completion_length': 206.5375, 'kl': 0.47359161376953124, 'epoch': 0.15}
{'loss': 0.0, 'grad_norm': 0.19115132093429565, 'learning_rate': 3.937499999999999e-06, 'rewards/reward_fn': 1.16992217451334, 'reward': 1.16992217451334, 'reward_std': 0.2609785484615713, 'completion_length': 208.775, 'kl': 0.4794609069824219, 'epoch': 0.15}
 94% 282/300 [1:56:43<07:25, 24.74s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 282)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.23388050496578217, 'learning_rate': 2.9999999999999992e-06, 'rewards/reward_fn': 1.116544672846794, 'reward': 1.116544672846794, 'reward_std': 0.28547151195816695, 'completion_length': 190.5375, 'kl': 0.5574325561523438, 'epoch': 0.15}
 96% 288/300 [1:59:16<05:06, 25.54s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 288)
    Unique invented tags: ['number']
    Occurrences: [('number', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1821039319038391, 'learning_rate': 2.0624999999999993e-06, 'rewards/reward_fn': 1.256972385942936, 'reward': 1.256972385942936, 'reward_std': 0.311754054017365, 'completion_length': 194.8375, 'kl': 0.4498313903808594, 'epoch': 0.16}
{'loss': 0.0, 'grad_norm': 0.16129879653453827, 'learning_rate': 1.1249999999999994e-06, 'rewards/reward_fn': 1.2160403043031693, 'reward': 1.2160403043031693, 'reward_std': 0.34193625347688794, 'completion_length': 179.6625, 'kl': 0.8539665222167969, 'epoch': 0.16}
{'loss': 0.0, 'grad_norm': 0.22020100057125092, 'learning_rate': 1.8749999999999934e-07, 'rewards/reward_fn': 1.2016037546098233, 'reward': 1.2016037546098233, 'reward_std': 0.3117403051815927, 'completion_length': 208.1625, 'kl': 0.6006492614746094, 'epoch': 0.16}
100% 300/300 [2:04:25<00:00, 24.97s/it][+] Step 300 → save_step_300.zip uploaded to Drive.
{'train_runtime': 7502.1292, 'train_samples_per_second': 0.16, 'train_steps_per_second': 0.04, 'train_loss': 9.973903426422718e-08, 'epoch': 0.16}
100% 300/300 [2:05:02<00:00, 25.01s/it]
[*] Saving final LoRA adapters to ./grpo_cot_resumed_final/final_lora...
[+] LF-GRPO Pipeline Execution Completed Successfully!
[rank0]:[W530 18:29:40.437973137 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
