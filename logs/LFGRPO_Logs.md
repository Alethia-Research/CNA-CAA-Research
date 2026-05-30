[+] Successfully wrote global sitecustomize.py to /usr/local/lib/python3.12/dist-packages/sitecustomize.py
[*] Environment check: Executing automated installation...
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24.9/24.9 MB 77.7 MB/s eta 0:00:00
[+] Installed uv package manager.
[*] Installing custom Unsloth and HF ecosystems...
[*] Installing NVIDIA nvjitlink for bitsandbytes...
[*] Installing vLLM from https://wheels.vllm.ai/cu128 matching CUDA 128...
[*] Pinning transformers>=4.51.3,<5.0 (vLLM 0.7.x + unsloth compat)...
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.0/44.0 kB 3.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 96.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 566.4/566.4 kB 43.4 MB/s eta 0:00:00
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
unsloth-zoo 2026.5.4 requires datasets!=4.0.*,!=4.1.0,<4.4.0,>=3.4.1, but you have datasets 4.8.5 which is incompatible.
unsloth-zoo 2026.5.4 requires trl!=0.19.0,<=0.24.0,>=0.18.2; sys_platform != "darwin" or platform_machine != "arm64", but you have trl 0.15.0.dev0 which is incompatible.
gradio 5.50.0 requires pandas<3.0,>=1.0, but you have pandas 3.0.3 which is incompatible.
gradio 5.50.0 requires starlette<1.0,>=0.40.0, but you have starlette 1.2.0 which is incompatible.
[+] Dependency installation complete!
[!] IMPORTANT: Please restart the Colab session (Runtime -> Restart session) to clear cached libraries.
[+] Restarting Python process after fresh install...
[+] Successfully wrote global sitecustomize.py to /usr/local/lib/python3.12/dist-packages/sitecustomize.py
2026-05-30 06:37:55.016791: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO 05-30 06:38:11 __init__.py:190] Automatically detected platform cuda.
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Unsloth: Could not find `steps_per_generation` in grpo_trainer
Unsloth: Could not find `generation_batch_size` in grpo_trainer
[+] Core machine learning environments detected.
[+] Post-restart: transformers version lock confirmed.
[+] Patched transformers quantizers/auto.py — torchao import is now guarded
[+] Patched vLLM inputs/registry.py ProcessorMixin import
[+] Patched vLLM multimodal/processing.py ProcessorMixin import
[+] Patched vLLM transformers_utils/processor.py
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
README.md: 7.93kB [00:00, 13.4MB/s]
main/train-00000-of-00001.parquet: 100% 2.31M/2.31M [00:01<00:00, 2.11MB/s]
main/test-00000-of-00001.parquet: 100% 419k/419k [00:00<00:00, 1.45MB/s]
Generating train split: 100% 7473/7473 [00:00<00:00, 130070.11 examples/s]
Generating test split: 100% 1319/1319 [00:00<00:00, 253809.56 examples/s]
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
model.safetensors: 100% 1.53G/1.53G [00:13<00:00, 112MB/s]
generation_config.json: 100% 270/270 [00:00<00:00, 2.24MB/s]
tokenizer_config.json: 7.36kB [00:00, 27.8MB/s]
vocab.json: 2.78MB [00:00, 112MB/s]
merges.txt: 1.67MB [00:00, 128MB/s]
added_tokens.json: 100% 605/605 [00:00<00:00, 5.29MB/s]
special_tokens_map.json: 100% 614/614 [00:00<00:00, 4.07MB/s]
tokenizer.json: 100% 11.4M/11.4M [00:00<00:00, 32.6MB/s]
unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit does not have a padding token! Will use pad_token = <|PAD_TOKEN|>.
[+] Spatial configuration: Freezing L0-L23. Adapting periphery layers: [24, 25, 26, 27]
Not an error, but Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters
are not enabled or a bias term (like in Qwen) is used.
Not an error, but Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters
are not enabled or a bias term (like in Qwen) is used.
Unsloth 2026.5.8 patched 28 layers with 28 QKV layers, 4 O layers and 4 MLP layers.
[*] Loading adapter weights from kridaydave/Qwen-1.5B-LFGRPO-OPTIM...
adapter_model.safetensors: 100% 21.1M/21.1M [00:04<00:00, 4.48MB/s]
[+] Adapter weights loaded — stage-1 weights restored, continuing from step 100.
Generating train split: 7473 examples [00:00, 307173.99 examples/s]
[*] GRPOConfig has no 'vllm_mode' param — skipping.
[*] GRPOConfig has no 'vllm_enforce_eager' param — skipping.
[*] vLLM fp16 inference model: Qwen/Qwen2.5-1.5B-Instruct
[+] Drive already mounted → /content/drive/MyDrive/grpo_checkpoints/grpo_cot_output
[*] vLLM: overriding model 'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit' → 'Qwen/Qwen2.5-1.5B-Instruct'
config.json: 100% 660/660 [00:00<00:00, 5.77MB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
WARNING 05-30 06:39:40 config.py:2386] Casting torch.bfloat16 to torch.float16.
INFO 05-30 06:39:59 config.py:542] This model supports multiple tasks: {'embed', 'classify', 'score', 'reward', 'generate'}. Defaulting to 'generate'.
INFO 05-30 06:39:59 llm_engine.py:234] Initializing a V0 LLM engine (v0.7.2) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda:0, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=Qwen/Qwen2.5-1.5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False, 
tokenizer_config.json: 7.30kB [00:00, 27.3MB/s]
vocab.json: 2.78MB [00:00, 92.8MB/s]
merges.txt: 1.67MB [00:00, 102MB/s]
tokenizer.json: 7.03MB [00:00, 142MB/s]
generation_config.json: 100% 242/242 [00:00<00:00, 1.69MB/s]
INFO 05-30 06:40:03 cuda.py:179] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 05-30 06:40:03 cuda.py:227] Using XFormers backend.
INFO 05-30 06:40:04 model_runner.py:1110] Starting to load model Qwen/Qwen2.5-1.5B-Instruct...
INFO 05-30 06:40:05 weight_utils.py:252] Using model weights format ['*.safetensors']
model.safetensors: 100% 3.09G/3.09G [00:34<00:00, 88.7MB/s]
INFO 05-30 06:40:40 weight_utils.py:297] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards: 100% 1/1 [00:07<00:00,  7.51s/it]
INFO 05-30 06:40:49 model_runner.py:1115] Loading model weights took 2.8884 GB
INFO 05-30 06:40:55 worker.py:267] Memory profiling takes 5.61 seconds
INFO 05-30 06:40:55 worker.py:267] the current vLLM instance can use total_gpu_memory (14.56GiB) x gpu_memory_utilization (0.50) = 7.28GiB
INFO 05-30 06:40:55 worker.py:267] model weights take 2.89GiB; non_torch_memory takes 0.01GiB; PyTorch activation peak memory takes 2.02GiB; the rest of the memory reserved for KV Cache is 2.36GiB.
INFO 05-30 06:40:55 executor_base.py:110] # CUDA blocks: 5517, # CPU blocks: 9362
INFO 05-30 06:40:55 executor_base.py:115] Maximum concurrency for 32768 tokens per request: 2.69x
INFO 05-30 06:41:01 model_runner.py:1434] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100% 35/35 [00:48<00:00,  1.39s/it]
INFO 05-30 06:41:50 model_runner.py:1562] Graph capturing finished in 49 secs, took 0.17 GiB
INFO 05-30 06:41:50 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 61.80 seconds
Fetching 1 files: 100% 1/1 [00:00<00:00, 8525.01it/s]
[+] Loaded fp16 base: 1 safetensors, 338 tensors
[+] Patched vLLM load_weights — LoRA merge from fp16 cache enabled.
[*] Launching LF-GRPO training loop for 300 steps...
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 7,473 | Num Epochs = 1 | Total steps = 300
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 4 x 1) = 4
 "-____-"     Trainable parameters = 5,275,648 of 1,548,989,952 (0.34% trained)
  0% 0/300 [00:00<?, ?it/s]
==================================================
[!] EMERGENT TAGS DETECTED (Step 0)
    Unique invented tags: ['calculate', 'reply']
    Occurrences: [('reply', 2), ('calculate', 1)]
==================================================
Unsloth: Will smartly offload gradients to save VRAM!

==================================================
GRADIENT INSULATION VERIFICATION REPORT
==================================================
[+] Total trainable parameters: 56
[*] Sample active parameter gradient norms:
  - base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight: grad_norm = 0.000118
  - base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weight: grad_norm = 0.005083
  - base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight: grad_norm = 0.000136
  - base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weight: grad_norm = 0.006319
  - base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weight: grad_norm = 0.000082
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
  0% 1/300 [00:30<2:32:54, 30.68s/it][*] vLLM sync #1: 28 LoRA-merged weights pushed (28 A / 28 B detected)

==================================================
[!] EMERGENT TAGS DETECTED (Step 1)
    Unique invented tags: ['p']
    Occurrences: [('p', 1)]
==================================================
  1% 2/300 [00:55<2:15:48, 27.34s/it][*] vLLM sync #2: 28 LoRA-merged weights pushed (28 A / 28 B detected)
  1% 3/300 [01:20<2:08:31, 25.96s/it][*] vLLM sync #3: 28 LoRA-merged weights pushed (28 A / 28 B detected)
  1% 4/300 [01:38<1:54:25, 23.19s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 4)
    Unique invented tags: ['math']
    Occurrences: [('math', 18)]
==================================================
{'loss': -0.0, 'grad_norm': 0.020806750282645226, 'learning_rate': 1.48e-05, 'rewards/reward_fn': 0.4735000032931566, 'reward': 0.4735000032931566, 'reward_std': 0.23546656146645545, 'completion_length': 233.4, 'kl': 3.457069396972656e-07, 'epoch': 0.0}
{'loss': 0.0, 'grad_norm': 0.039073750376701355, 'learning_rate': 1.455e-05, 'rewards/reward_fn': 0.8232500046491623, 'reward': 0.8232500046491623, 'reward_std': 0.2103642667643726, 'completion_length': 225.975, 'kl': 5.4925680160522464e-06, 'epoch': 0.01}
  3% 10/300 [03:54<1:46:37, 22.06s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 10)
    Unique invented tags: ['nothink']
    Occurrences: [('nothink', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.044132500886917114, 'learning_rate': 1.43e-05, 'rewards/reward_fn': 0.7239245237782598, 'reward': 0.7239245237782598, 'reward_std': 0.2568286551162601, 'completion_length': 217.775, 'kl': 7.43865966796875e-06, 'epoch': 0.01}
  6% 17/300 [06:42<1:52:49, 23.92s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 17)
    Unique invented tags: ['bold']
    Occurrences: [('bold', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.07171975076198578, 'learning_rate': 1.405e-05, 'rewards/reward_fn': 0.7605000036768615, 'reward': 0.7605000036768615, 'reward_std': 0.22980970442295073, 'completion_length': 235.45, 'kl': 9.754300117492676e-06, 'epoch': 0.01}
{'loss': 0.0, 'grad_norm': 0.05839924141764641, 'learning_rate': 1.3800000000000002e-05, 'rewards/reward_fn': 0.7055298069491982, 'reward': 0.7055298069491982, 'reward_std': 0.2651650466024876, 'completion_length': 232.95, 'kl': 2.0119547843933106e-05, 'epoch': 0.01}
  9% 27/300 [10:35<1:48:25, 23.83s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 27)
    Unique invented tags: ['nowalk']
    Occurrences: [('nowalk', 10)]
==================================================
{'loss': -0.0, 'grad_norm': 0.08377577364444733, 'learning_rate': 1.355e-05, 'rewards/reward_fn': 0.6290000043809414, 'reward': 0.6290000043809414, 'reward_std': 0.169705630838871, 'completion_length': 232.825, 'kl': 5.779266357421875e-05, 'epoch': 0.02}
 11% 32/300 [12:25<1:41:52, 22.81s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 32)
    Unique invented tags: ['p']
    Occurrences: [('p', 11)]
==================================================
{'loss': -0.0, 'grad_norm': 0.06703626364469528, 'learning_rate': 1.3300000000000001e-05, 'rewards/reward_fn': 0.5990000008605421, 'reward': 0.5990000008605421, 'reward_std': 0.19657568661496044, 'completion_length': 198.2, 'kl': 0.00012102127075195313, 'epoch': 0.02}
 12% 36/300 [13:43<1:28:17, 20.07s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 36)
    Unique invented tags: ['number']
    Occurrences: [('number', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.06083919107913971, 'learning_rate': 1.305e-05, 'rewards/reward_fn': 0.5160000022500754, 'reward': 0.5160000022500754, 'reward_std': 0.340825467184186, 'completion_length': 232.675, 'kl': 0.0003002792596817017, 'epoch': 0.02}
 15% 44/300 [16:47<1:33:47, 21.98s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 44)
    Unique invented tags: ['1', '2', '3', 'p']
    Occurrences: [('p', 3), ('1', 1), ('2', 1), ('3', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.02407451532781124, 'learning_rate': 1.2800000000000001e-05, 'rewards/reward_fn': 0.5670000045560301, 'reward': 0.5670000045560301, 'reward_std': 0.2177888922393322, 'completion_length': 215.425, 'kl': 0.000524023175239563, 'epoch': 0.02}
{'loss': -0.0, 'grad_norm': 0.02626323141157627, 'learning_rate': 1.255e-05, 'rewards/reward_fn': 0.5614406570792199, 'reward': 0.5614406570792199, 'reward_std': 0.21797588700428605, 'completion_length': 222.375, 'kl': 0.00033744871616363524, 'epoch': 0.03}
 17% 51/300 [19:05<1:24:28, 20.36s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 51)
    Unique invented tags: ['thinking']
    Occurrences: [('thinking', 2)]
==================================================
 18% 54/300 [20:12<1:25:16, 20.80s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 54)
    Unique invented tags: ['li', 'p', 'ul']
    Occurrences: [('li', 6), ('ul', 2), ('p', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.04528878629207611, 'learning_rate': 1.2299999999999999e-05, 'rewards/reward_fn': 0.6270025009289384, 'reward': 0.6270025009289384, 'reward_std': 0.29557416774332523, 'completion_length': 202.825, 'kl': 0.00034820437431335447, 'epoch': 0.03}
 19% 58/300 [21:38<1:26:11, 21.37s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 58)
    Unique invented tags: ['result']
    Occurrences: [('result', 4)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 58)
    Unique invented tags: ['br']
    Occurrences: [('br', 5)]
==================================================
 20% 59/300 [21:56<1:21:05, 20.19s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 59)
    Unique invented tags: ['noinput']
    Occurrences: [('noinput', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.05099014192819595, 'learning_rate': 1.205e-05, 'rewards/reward_fn': 0.6137525065802037, 'reward': 0.6137525065802037, 'reward_std': 0.3843160753138363, 'completion_length': 208.475, 'kl': 0.00106145441532135, 'epoch': 0.03}
{'loss': -0.0, 'grad_norm': 0.06601651012897491, 'learning_rate': 1.1799999999999999e-05, 'rewards/reward_fn': 0.5497500003315509, 'reward': 0.5497500003315509, 'reward_std': 0.38643385358154775, 'completion_length': 241.775, 'kl': 0.0006718724966049194, 'epoch': 0.03}
 23% 69/300 [25:52<1:26:31, 22.48s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 69)
    Unique invented tags: ['math']
    Occurrences: [('math', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.10385186225175858, 'learning_rate': 1.1550000000000001e-05, 'rewards/reward_fn': 0.7484999999403954, 'reward': 0.7484999999403954, 'reward_std': 0.21849599573761225, 'completion_length': 208.95, 'kl': 0.0012143313884735107, 'epoch': 0.04}
 24% 72/300 [26:57<1:24:34, 22.26s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 72)
    Unique invented tags: ['endif']
    Occurrences: [('endif', 3)]
==================================================
 24% 73/300 [27:13<1:16:31, 20.22s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 73)
    Unique invented tags: ['answer', 'calculate']
    Occurrences: [('calculate', 2), ('answer', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 73)
    Unique invented tags: ['p']
    Occurrences: [('p', 1)]
==================================================
 25% 74/300 [27:41<1:25:25, 22.68s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 74)
    Unique invented tags: ['br']
    Occurrences: [('br', 7)]
==================================================
{'loss': 0.0, 'grad_norm': 0.02929769828915596, 'learning_rate': 1.13e-05, 'rewards/reward_fn': 0.625000000372529, 'reward': 0.625000000372529, 'reward_std': 0.23334523849189281, 'completion_length': 240.1, 'kl': 0.0005483955144882202, 'epoch': 0.04}
 25% 75/300 [28:07<1:28:04, 23.49s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 75)
    Unique invented tags: ['br', 'strong']
    Occurrences: [('strong', 8), ('br', 2)]
==================================================
 26% 78/300 [29:18<1:27:56, 23.77s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 78)
    Unique invented tags: ['finish']
    Occurrences: [('finish', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.023376859724521637, 'learning_rate': 1.1050000000000001e-05, 'rewards/reward_fn': 0.6127500051632524, 'reward': 0.6127500051632524, 'reward_std': 0.35602826848626135, 'completion_length': 246.0, 'kl': 0.0007265061140060425, 'epoch': 0.04}
{'loss': -0.0, 'grad_norm': 0.043770983815193176, 'learning_rate': 1.08e-05, 'rewards/reward_fn': 0.40950250178575515, 'reward': 0.40950250178575515, 'reward_std': 0.2609259381890297, 'completion_length': 232.75, 'kl': 0.0010673940181732179, 'epoch': 0.05}
 28% 85/300 [32:07<1:26:57, 24.27s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 85)
    Unique invented tags: ['math']
    Occurrences: [('math', 6)]
==================================================
{'loss': -0.0, 'grad_norm': 0.07126773148775101, 'learning_rate': 1.055e-05, 'rewards/reward_fn': 0.5800000049173832, 'reward': 0.5800000049173832, 'reward_std': 0.3648671008646488, 'completion_length': 226.05, 'kl': 0.0009352028369903565, 'epoch': 0.05}
 31% 93/300 [35:05<1:15:24, 21.86s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 93)
    Unique invented tags: ['endif']
    Occurrences: [('endif', 2)]
==================================================
 31% 94/300 [35:29<1:17:10, 22.48s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 94)
    Unique invented tags: ['math', 'underline']
    Occurrences: [('math', 6), ('underline', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.053968824446201324, 'learning_rate': 1.03e-05, 'rewards/reward_fn': 0.7107500094920397, 'reward': 0.7107500094920397, 'reward_std': 0.29521709010005, 'completion_length': 218.55, 'kl': 0.0010063707828521728, 'epoch': 0.05}
 32% 95/300 [35:53<1:18:35, 23.00s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 95)
    Unique invented tags: ['do']
    Occurrences: [('do', 4)]
==================================================
{'loss': 0.0, 'grad_norm': 0.1630244106054306, 'learning_rate': 1.005e-05, 'rewards/reward_fn': 0.6217500043101609, 'reward': 0.6217500043101609, 'reward_std': 0.16157390316948295, 'completion_length': 224.425, 'kl': 0.00140572190284729, 'epoch': 0.05}
 33% 100/300 [37:45<1:15:04, 22.52s/it][+] Step 100 → save_step_100.zip uploaded to Drive.
{'loss': -0.0, 'grad_norm': 0.07375781983137131, 'learning_rate': 9.8e-06, 'rewards/reward_fn': 0.6390000009909272, 'reward': 0.6390000009909272, 'reward_std': 0.26304372139275073, 'completion_length': 221.25, 'kl': 0.0005045294761657715, 'epoch': 0.06}
 36% 108/300 [40:47<1:14:21, 23.24s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 108)
    Unique invented tags: ['mathematically']
    Occurrences: [('mathematically', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.05871136859059334, 'learning_rate': 9.550000000000002e-06, 'rewards/reward_fn': 0.46000000089406967, 'reward': 0.46000000089406967, 'reward_std': 0.34648232338950036, 'completion_length': 244.1, 'kl': nan, 'epoch': 0.06}
{'loss': -0.0, 'grad_norm': 0.06454616039991379, 'learning_rate': 9.3e-06, 'rewards/reward_fn': 0.7359999984502792, 'reward': 0.7359999984502792, 'reward_std': 0.17253405563533306, 'completion_length': 229.25, 'kl': 0.0007079988718032837, 'epoch': 0.06}
{'loss': 0.0, 'grad_norm': 0.052154846489429474, 'learning_rate': 9.050000000000001e-06, 'rewards/reward_fn': 0.6120000032708048, 'reward': 0.6120000032708048, 'reward_std': 0.3104198770597577, 'completion_length': 224.5, 'kl': 0.0004996359348297119, 'epoch': 0.06}
 41% 122/300 [46:08<1:10:48, 23.87s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 122)
    Unique invented tags: ['math']
    Occurrences: [('math', 8)]
==================================================
 41% 123/300 [46:35<1:12:48, 24.68s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 123)
    Unique invented tags: ['nums']
    Occurrences: [('nums', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.05784795060753822, 'learning_rate': 8.8e-06, 'rewards/reward_fn': 0.521500005479902, 'reward': 0.521500005479902, 'reward_std': 0.26657925862818954, 'completion_length': 229.625, 'kl': 0.0014080971479415894, 'epoch': 0.07}
{'loss': -0.0, 'grad_norm': 0.07105469703674316, 'learning_rate': 8.55e-06, 'rewards/reward_fn': 0.5955000036396086, 'reward': 0.5955000036396086, 'reward_std': 0.28213560972362756, 'completion_length': 244.95, 'kl': 0.0007816940546035766, 'epoch': 0.07}
{'loss': -0.0, 'grad_norm': 0.041369061917066574, 'learning_rate': 8.3e-06, 'rewards/reward_fn': 0.7632500039413571, 'reward': 0.7632500039413571, 'reward_std': 0.28602469731122254, 'completion_length': 226.0, 'kl': 0.0006996840238571167, 'epoch': 0.07}
{'loss': 0.0, 'grad_norm': 0.056826476007699966, 'learning_rate': 8.05e-06, 'rewards/reward_fn': 0.7300000037997961, 'reward': 0.7300000037997961, 'reward_std': 0.20435386030003427, 'completion_length': 217.0, 'kl': 0.0007770180702209473, 'epoch': 0.07}
 47% 141/300 [53:23<57:06, 21.55s/it]  
==================================================
[!] EMERGENT TAGS DETECTED (Step 141)
    Unique invented tags: ['br']
    Occurrences: [('br', 4)]
==================================================
{'loss': -0.0, 'grad_norm': 0.041425932198762894, 'learning_rate': 7.8e-06, 'rewards/reward_fn': 0.6855000078678131, 'reward': 0.6855000078678131, 'reward_std': 0.34577522799372673, 'completion_length': 207.0, 'kl': 0.001402100920677185, 'epoch': 0.08}
 50% 149/300 [56:08<51:56, 20.64s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 149)
    Unique invented tags: ['yes']
    Occurrences: [('yes', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.058821119368076324, 'learning_rate': 7.55e-06, 'rewards/reward_fn': 0.638500005658716, 'reward': 0.638500005658716, 'reward_std': 0.22980971038341522, 'completion_length': 204.95, 'kl': 0.0013457238674163818, 'epoch': 0.08}
 51% 153/300 [57:33<48:56, 19.97s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 153)
    Unique invented tags: ['br']
    Occurrences: [('br', 5)]
==================================================
{'loss': -0.0, 'grad_norm': 0.05437970533967018, 'learning_rate': 7.3e-06, 'rewards/reward_fn': 0.5237500067800284, 'reward': 0.5237500067800284, 'reward_std': 0.200464775133878, 'completion_length': 211.975, 'kl': 0.001434844732284546, 'epoch': 0.08}
 52% 157/300 [59:12<56:56, 23.89s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 157)
    Unique invented tags: ['prev']
    Occurrences: [('prev', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 7.049999999999999e-06, 'rewards/reward_fn': 0.49100000225007534, 'reward': 0.49100000225007534, 'reward_std': 0.22203153148293495, 'completion_length': 212.85, 'kl': 0.0017145603895187377, 'epoch': 0.09}
 53% 160/300 [1:00:15<51:33, 22.10s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 160)
    Unique invented tags: ['nowkat']
    Occurrences: [('nowkat', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.0673796758055687, 'learning_rate': 6.8e-06, 'rewards/reward_fn': 0.5663214886561037, 'reward': 0.5663214886561037, 'reward_std': 0.34852647385559976, 'completion_length': 199.55, 'kl': 0.001896345615386963, 'epoch': 0.09}
 55% 166/300 [1:02:23<49:34, 22.20s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 166)
    Unique invented tags: ['Finally']
    Occurrences: [('Finally', 1)]
==================================================
 56% 169/300 [1:03:28<46:53, 21.48s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 169)
    Unique invented tags: ['image']
    Occurrences: [('image', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.05722348019480705, 'learning_rate': 6.55e-06, 'rewards/reward_fn': 0.6317822261713445, 'reward': 0.6317822261713445, 'reward_std': 0.3207185100764036, 'completion_length': 228.175, 'kl': 0.0022719651460647583, 'epoch': 0.09}
 58% 174/300 [1:05:15<43:31, 20.72s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 174)
    Unique invented tags: ['60']
    Occurrences: [('60', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.01726868934929371, 'learning_rate': 6.3e-06, 'rewards/reward_fn': 0.55800000121817, 'reward': 0.55800000121817, 'reward_std': 0.2588010862469673, 'completion_length': 216.25, 'kl': 0.0015424489974975586, 'epoch': 0.09}
{'loss': 0.0, 'grad_norm': 0.06068224459886551, 'learning_rate': 6.05e-06, 'rewards/reward_fn': 0.5780000039376318, 'reward': 0.5780000039376318, 'reward_std': 0.36769552864134314, 'completion_length': 199.725, 'kl': 0.0011869549751281738, 'epoch': 0.1}
 61% 184/300 [1:08:54<43:47, 22.65s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 184)
    Unique invented tags: ['thinking']
    Occurrences: [('thinking', 4)]
==================================================
{'loss': -0.0, 'grad_norm': 0.07771603763103485, 'learning_rate': 5.8e-06, 'rewards/reward_fn': 0.4095000034198165, 'reward': 0.4095000034198165, 'reward_std': 0.207182290032506, 'completion_length': 225.875, 'kl': 0.0010964691638946534, 'epoch': 0.1}
 63% 189/300 [1:10:52<44:10, 23.88s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 189)
    Unique invented tags: ['step']
    Occurrences: [('step', 3)]
==================================================
{'loss': 0.0, 'grad_norm': 0.07449077069759369, 'learning_rate': 5.55e-06, 'rewards/reward_fn': 0.5377500033937395, 'reward': 0.5377500033937395, 'reward_std': 0.22379929777234792, 'completion_length': 229.35, 'kl': 0.0010374456644058228, 'epoch': 0.1}
 64% 191/300 [1:11:44<45:53, 25.26s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 191)
    Unique invented tags: ['calculate']
    Occurrences: [('calculate', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.07436114549636841, 'learning_rate': 5.3e-06, 'rewards/reward_fn': 0.7080000013113021, 'reward': 0.7080000013113021, 'reward_std': 0.33092597462236883, 'completion_length': 229.65, 'kl': 0.001393383741378784, 'epoch': 0.1}
 65% 195/300 [1:13:10<39:19, 22.47s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 195)
    Unique invented tags: ['answer']
    Occurrences: [('answer', 2)]
==================================================
 65% 196/300 [1:13:28<36:24, 21.00s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 196)
    Unique invented tags: ['br', 'div']
    Occurrences: [('div', 2), ('br', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 5.05e-06, 'rewards/reward_fn': 0.8313019055873155, 'reward': 0.8313019055873155, 'reward_std': 0.20679747494868933, 'completion_length': 194.25, 'kl': 0.0010110646486282348, 'epoch': 0.11}
 67% 200/300 [1:14:52<34:20, 20.60s/it][+] Step 200 → save_step_200.zip uploaded to Drive.

==================================================
[!] EMERGENT TAGS DETECTED (Step 200)
    Unique invented tags: ['math']
    Occurrences: [('math', 8)]
==================================================
{'loss': 0.0, 'grad_norm': 0.04661231487989426, 'learning_rate': 4.800000000000001e-06, 'rewards/reward_fn': 0.5655000049620866, 'reward': 0.5655000049620866, 'reward_std': 0.19445436578243971, 'completion_length': 216.325, 'kl': 0.0011232048273086548, 'epoch': 0.11}
 70% 209/300 [1:18:11<31:49, 20.99s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 209)
    Unique invented tags: ['finally']
    Occurrences: [('finally', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.045540571212768555, 'learning_rate': 4.5500000000000005e-06, 'rewards/reward_fn': 0.6050000049173831, 'reward': 0.6050000049173831, 'reward_std': 0.2474873773753643, 'completion_length': 216.675, 'kl': 0.0010109573602676391, 'epoch': 0.11}
 70% 210/300 [1:18:32<31:12, 20.81s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 210)
    Unique invented tags: ['write']
    Occurrences: [('write', 3)]
==================================================
 71% 214/300 [1:19:59<31:14, 21.80s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 214)
    Unique invented tags: ['finish']
    Occurrences: [('finish', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.07616405934095383, 'learning_rate': 4.3e-06, 'rewards/reward_fn': 0.7620000045746564, 'reward': 0.7620000045746564, 'reward_std': 0.27577164778485896, 'completion_length': 220.35, 'kl': 0.0011192053556442262, 'epoch': 0.12}
 72% 217/300 [1:21:08<31:04, 22.46s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 217)
    Unique invented tags: ['math']
    Occurrences: [('math', 2)]
==================================================
 73% 219/300 [1:21:53<30:14, 22.40s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 219)
    Unique invented tags: ['br', 'div']
    Occurrences: [('br', 7), ('div', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 219)
    Unique invented tags: ['underline']
    Occurrences: [('underline', 8)]
==================================================
{'loss': -0.0, 'grad_norm': 0.058042462915182114, 'learning_rate': 4.05e-06, 'rewards/reward_fn': 0.7207525063306093, 'reward': 0.7207525063306093, 'reward_std': 0.27966427206993105, 'completion_length': 205.45, 'kl': 0.0018613606691360474, 'epoch': 0.12}
 74% 222/300 [1:22:59<29:03, 22.36s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 222)
    Unique invented tags: ['finish']
    Occurrences: [('finish', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.06801819801330566, 'learning_rate': 3.8000000000000005e-06, 'rewards/reward_fn': 0.5355000051669776, 'reward': 0.5355000051669776, 'reward_std': 0.3075914531946182, 'completion_length': 248.275, 'kl': 0.0011604815721511842, 'epoch': 0.12}
{'loss': -0.0, 'grad_norm': 0.07128676027059555, 'learning_rate': 3.55e-06, 'rewards/reward_fn': 0.5257499992847443, 'reward': 0.5257499992847443, 'reward_std': 0.22026376575231552, 'completion_length': 219.325, 'kl': 0.0011421829462051392, 'epoch': 0.12}
 77% 231/300 [1:26:27<24:52, 21.62s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 231)
    Unique invented tags: ['final_answer']
    Occurrences: [('final_answer', 2)]
==================================================
 77% 232/300 [1:26:50<24:48, 21.89s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 232)
    Unique invented tags: ['math']
    Occurrences: [('math', 4)]
==================================================
{'loss': -0.0, 'grad_norm': 0.042771246284246445, 'learning_rate': 3.3e-06, 'rewards/reward_fn': 0.5845074995420874, 'reward': 0.5845074995420874, 'reward_std': 0.2425411651842296, 'completion_length': 218.25, 'kl': 0.0013555943965911864, 'epoch': 0.13}
{'loss': -0.0, 'grad_norm': 0.0, 'learning_rate': 3.05e-06, 'rewards/reward_fn': 0.7045880837365985, 'reward': 0.7045880837365985, 'reward_std': 0.19376874305307865, 'completion_length': 217.175, 'kl': 0.0009859919548034668, 'epoch': 0.13}
 81% 244/300 [1:31:15<20:11, 21.63s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 244)
    Unique invented tags: ['finish']
    Occurrences: [('finish', 1)]
==================================================
{'loss': 0.0, 'grad_norm': 0.03900090605020523, 'learning_rate': 2.8000000000000003e-06, 'rewards/reward_fn': 0.7262500047683715, 'reward': 0.7262500047683715, 'reward_std': 0.22026376267895104, 'completion_length': 244.15, 'kl': 0.0009435832500457764, 'epoch': 0.13}
 83% 249/300 [1:33:20<20:27, 24.06s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 249)
    Unique invented tags: ['math']
    Occurrences: [('math', 16)]
==================================================
{'loss': -0.0, 'grad_norm': 0.07309787720441818, 'learning_rate': 2.55e-06, 'rewards/reward_fn': 0.6575000049546361, 'reward': 0.6575000049546361, 'reward_std': 0.2793071810156107, 'completion_length': 244.35, 'kl': 0.0006178349256515502, 'epoch': 0.13}
 84% 252/300 [1:34:27<18:30, 23.13s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 252)
    Unique invented tags: ['result']
    Occurrences: [('result', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 252)
    Unique invented tags: ['math']
    Occurrences: [('math', 2)]
==================================================
{'loss': -0.0, 'grad_norm': 0.030148839578032494, 'learning_rate': 2.3e-06, 'rewards/reward_fn': 0.607218649238348, 'reward': 0.607218649238348, 'reward_std': 0.18698540180921555, 'completion_length': 222.425, 'kl': 0.0008812427520751953, 'epoch': 0.14}
 85% 255/300 [1:35:32<17:02, 22.72s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 255)
    Unique invented tags: ['calculate']
    Occurrences: [('calculate', 2)]
==================================================
 86% 258/300 [1:36:36<14:51, 21.23s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 258)
    Unique invented tags: ['answer', 'nowayout']
    Occurrences: [('nowayout', 2), ('answer', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.05847017839550972, 'learning_rate': 2.05e-06, 'rewards/reward_fn': 0.6140000056475401, 'reward': 0.6140000056475401, 'reward_std': 0.27718586213886737, 'completion_length': 210.95, 'kl': 0.0018019795417785645, 'epoch': 0.14}
{'loss': -0.0, 'grad_norm': 0.06754658371210098, 'learning_rate': 1.8e-06, 'rewards/reward_fn': 0.7010000044479966, 'reward': 0.7010000044479966, 'reward_std': 0.3733523812144995, 'completion_length': 203.375, 'kl': 0.0018405795097351074, 'epoch': 0.14}
 89% 268/300 [1:40:24<12:40, 23.75s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 268)
    Unique invented tags: ['show']
    Occurrences: [('show', 8)]
==================================================
 90% 269/300 [1:40:48<12:18, 23.82s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 269)
    Unique invented tags: ['underline']
    Occurrences: [('underline', 6)]
==================================================
{'loss': 0.0, 'grad_norm': 0.06377393752336502, 'learning_rate': 1.55e-06, 'rewards/reward_fn': 0.5702500033192337, 'reward': 0.5702500033192337, 'reward_std': 0.4133039128035307, 'completion_length': 243.075, 'kl': 0.001459646224975586, 'epoch': 0.14}
 90% 270/300 [1:41:14<12:17, 24.57s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 270)
    Unique invented tags: ['li', 'strong', 'ul']
    Occurrences: [('strong', 8), ('li', 4), ('ul', 2)]
==================================================
 91% 272/300 [1:42:00<11:08, 23.86s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 272)
    Unique invented tags: ['develop']
    Occurrences: [('develop', 1)]
==================================================
{'loss': -0.0, 'grad_norm': 0.07045049965381622, 'learning_rate': 1.3e-06, 'rewards/reward_fn': 0.41775000458583234, 'reward': 0.41775000458583234, 'reward_std': 0.3609780135564506, 'completion_length': 225.875, 'kl': 0.0015735894441604615, 'epoch': 0.15}
 92% 276/300 [1:43:34<09:21, 23.39s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 276)
    Unique invented tags: ['calculate', 'output']
    Occurrences: [('calculate', 6), ('output', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.0570988692343235, 'learning_rate': 1.0500000000000001e-06, 'rewards/reward_fn': 0.684500005748123, 'reward': 0.684500005748123, 'reward_std': 0.3090056672692299, 'completion_length': 232.65, 'kl': 0.0013924449682235717, 'epoch': 0.15}
 95% 284/300 [1:46:53<06:48, 25.52s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 284)
    Unique invented tags: ['br']
    Occurrences: [('br', 5)]
==================================================
{'loss': 0.0, 'grad_norm': 0.07168667763471603, 'learning_rate': 8.000000000000001e-07, 'rewards/reward_fn': 0.5059999977238476, 'reward': 0.5059999977238476, 'reward_std': 0.3196122642606497, 'completion_length': 241.8, 'kl': 0.0018512904644012451, 'epoch': 0.15}
 95% 285/300 [1:47:17<06:15, 25.05s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 285)
    Unique invented tags: ['result']
    Occurrences: [('result', 2)]
==================================================

==================================================
[!] EMERGENT TAGS DETECTED (Step 285)
    Unique invented tags: ['underline']
    Occurrences: [('underline', 2)]
==================================================
 96% 288/300 [1:48:23<04:34, 22.84s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 288)
    Unique invented tags: ['COMPUTE']
    Occurrences: [('COMPUTE', 2)]
==================================================
 96% 289/300 [1:48:43<04:03, 22.12s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 289)
    Unique invented tags: ['math']
    Occurrences: [('math', 8)]
==================================================
{'loss': -0.0, 'grad_norm': 0.10557746887207031, 'learning_rate': 5.5e-07, 'rewards/reward_fn': 0.5397500034421683, 'reward': 0.5397500034421683, 'reward_std': 0.3136018618941307, 'completion_length': 203.7, 'kl': 0.0016257643699645996, 'epoch': 0.16}
 97% 291/300 [1:49:23<03:07, 20.82s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 291)
    Unique invented tags: ['finish']
    Occurrences: [('finish', 2)]
==================================================
 97% 292/300 [1:49:40<02:36, 19.57s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 292)
    Unique invented tags: ['math']
    Occurrences: [('math', 2)]
==================================================
{'loss': 0.0, 'grad_norm': 0.08613526076078415, 'learning_rate': 3.0000000000000004e-07, 'rewards/reward_fn': 0.5990000100806355, 'reward': 0.5990000100806355, 'reward_std': 0.23193103224039077, 'completion_length': 189.475, 'kl': 0.0027889788150787355, 'epoch': 0.16}
 99% 297/300 [1:51:36<01:10, 23.39s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 297)
    Unique invented tags: ['math']
    Occurrences: [('math', 12)]
==================================================
 99% 298/300 [1:51:52<00:42, 21.26s/it]
==================================================
[!] EMERGENT TAGS DETECTED (Step 298)
    Unique invented tags: ['br']
    Occurrences: [('br', 13)]
==================================================
{'loss': -0.0, 'grad_norm': 0.03338576853275299, 'learning_rate': 5.0000000000000004e-08, 'rewards/reward_fn': 0.6030024980194867, 'reward': 0.6030024980194867, 'reward_std': 0.22203506268560885, 'completion_length': 215.525, 'kl': 0.001299351453781128, 'epoch': 0.16}
100% 300/300 [1:52:31<00:00, 20.29s/it][+] Step 300 → save_step_300.zip uploaded to Drive.
{'train_runtime': 6756.0271, 'train_samples_per_second': 0.178, 'train_steps_per_second': 0.044, 'train_loss': 1.3162692387898763e-09, 'epoch': 0.16}
100% 300/300 [1:52:36<00:00, 22.52s/it]
[*] Saving final LoRA adapters to ./grpo_cot_output/final_lora...
[+] LF-GRPO Pipeline Execution Completed Successfully!
[rank0]:[W530 08:34:35.154236112 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
