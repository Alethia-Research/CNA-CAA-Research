🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
[unsloth.import_fixes|WARNING]Unsloth: vLLM was built for CUDA 13 but this system has CUDA 12.8. Please reinstall vLLM with the correct CUDA version:

  uv pip install https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cu128-cp38-abi3-manylinux_2_35_x86_64.whl
🦥 Unsloth Zoo will now patch everything to make training faster!
[*] Starting GRPO training in mode: STEP-GRPO
[*] Model target: unsloth/Qwen2.5-1.5B-Instruct
[*] Dataset: data/gsm8k_train_grpo.jsonl
[*] Output directory: ./grpo_cot_output
[+] Device detected: CUDA. Using vLLM: False
[*] Loading model and tokenizer...
==((====))==  Unsloth 2026.5.8: Fast Qwen2 patching. Transformers: 5.5.0. vLLM: 0.21.0.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
model.safetensors: 100% 1.53G/1.53G [00:17<00:00, 88.9MB/s]
Loading weights: 100% 338/338 [00:01<00:00, 212.17it/s]
generation_config.json: 100% 270/270 [00:00<00:00, 1.62MB/s]
config.json: 100% 1.58k/1.58k [00:00<00:00, 5.08MB/s]
tokenizer_config.json: 100% 7.36k/7.36k [00:00<00:00, 21.4MB/s]
vocab.json: 100% 2.78M/2.78M [00:00<00:00, 37.1MB/s]
merges.txt: 100% 1.67M/1.67M [00:00<00:00, 109MB/s]
tokenizer.json: 100% 11.4M/11.4M [00:00<00:00, 25.1MB/s]
added_tokens.json: 100% 605/605 [00:00<00:00, 4.17MB/s]
special_tokens_map.json: 100% 614/614 [00:00<00:00, 4.44MB/s]
unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit does not have a padding token! Will use pad_token = <|PAD_TOKEN|>.
[+] Dynamic layer configuration resolved preset 'last_4' to layers: [24, 25, 26, 27]
[*] Applying PEFT LoRA adapters...
Not an error, but Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters
are not enabled or a bias term (like in Qwen) is used.
Not an error, but Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters
are not enabled or a bias term (like in Qwen) is used.
Unsloth 2026.5.8 patched 28 layers with 28 QKV layers, 4 O layers and 4 MLP layers.
[*] Loading training dataset from data/gsm8k_train_grpo.jsonl...
Generating train split: 1000 examples [00:00, 61641.06 examples/s]
[*] Launching training loop...
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None}.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 1,000 | Num Epochs = 1 | Total steps = 5
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 4 x 1) = 4
 "-____-"     Trainable parameters = 5,275,648 of 1,548,989,952 (0.34% trained)
  0% 0/5 [00:00<?, ?it/s]Passing `generation_config` together with generation-related arguments=({'pad_token_id', 'disable_compile', 'cache_implementation'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
Both `max_new_tokens` (=384) and `max_length`(=32768) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:71: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:281: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:71: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:281: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
Unsloth: Will smartly offload gradients to save VRAM!

==================================================
GRADIENT INSULATION VERIFICATION REPORT
==================================================
[+] Total trainable parameters: 56
[*] Sample active parameter gradient norms:
  - base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weight: grad_norm = 0.002211
  - base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weight: grad_norm = 0.003018
  - base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weight: grad_norm = 0.000000
[+] Total frozen parameters: 338
[*] Sample frozen parameter gradient norms:
  - base_model.model.model.embed_tokens.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.bias: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.bias: grad_norm = 0.000000
[+] VERIFICATION SUCCESS: 100% gradient insulation confirmed! All frozen layers have 0.0 gradient norm.
==================================================


==================================================
GRADIENT INSULATION VERIFICATION REPORT
==================================================
[+] Total trainable parameters: 56
[*] Sample active parameter gradient norms:
  - base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weight: grad_norm = 0.002593
  - base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weight: grad_norm = 0.003460
  - base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weight: grad_norm = 0.000000
[+] Total frozen parameters: 338
[*] Sample frozen parameter gradient norms:
  - base_model.model.model.embed_tokens.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.bias: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.bias: grad_norm = 0.000000
[+] VERIFICATION SUCCESS: 100% gradient insulation confirmed! All frozen layers have 0.0 gradient norm.
==================================================


==================================================
GRADIENT INSULATION VERIFICATION REPORT
==================================================
[+] Total trainable parameters: 56
[*] Sample active parameter gradient norms:
  - base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weight: grad_norm = 0.003289
  - base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight: grad_norm = 0.000000
  - base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weight: grad_norm = 0.004288
  - base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weight: grad_norm = 0.000000
[+] Total frozen parameters: 338
[*] Sample frozen parameter gradient norms:
  - base_model.model.model.embed_tokens.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.q_proj.bias: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.weight: grad_norm = 0.000000
  - base_model.model.model.layers.0.self_attn.k_proj.bias: grad_norm = 0.000000
[+] VERIFICATION SUCCESS: 100% gradient insulation confirmed! All frozen layers have 0.0 gradient norm.
==================================================

 20% 1/5 [01:04<04:17, 64.34s/it]Both `max_new_tokens` (=384) and `max_length`(=32768) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:71: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:281: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:71: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:281: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.
 40% 2/5 [01:23<01:53, 37.96s/it]Both `max_new_tokens` (=384) and `max_length`(=32768) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
 60% 3/5 [01:47<01:03, 31.64s/it]Both `max_new_tokens` (=384) and `max_length`(=32768) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
 80% 4/5 [01:59<00:23, 23.59s/it]Both `max_new_tokens` (=384) and `max_length`(=32768) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
{'loss': '5.96e-09', 'grad_norm': '0.08729', 'learning_rate': '5e-06', 'num_tokens': '6933', 'completions/mean_length': '230.2', 'completions/min_length': '171', 'completions/max_length': '285.8', 'completions/clipped_ratio': '0.05', 'completions/mean_terminated_length': '226.3', 'completions/min_terminated_length': '171', 'completions/max_terminated_length': '284.8', 'rewards/reward_fn/mean': '0.065', 'rewards/reward_fn/std': '0.02517', 'reward': '0.065', 'reward_std': '0.02517', 'frac_reward_zero_std': '0.8', 'completion_length': '230.2', 'kl': '9.845e-06', 'clip_ratio/low_mean': '0', 'clip_ratio/low_min': '0', 'clip_ratio/high_mean': '0', 'clip_ratio/high_max': '0', 'clip_ratio/region_mean': '0', 'epoch': '0.005'}
100% 5/5 [02:28<00:00, 25.52s/it]Unsloth: Restored added_tokens_decoder metadata in ./grpo_cot_output/checkpoint-5/tokenizer_config.json.
{'train_runtime': '148.9', 'train_samples_per_second': '0.134', 'train_steps_per_second': '0.034', 'train_loss': '5.96e-09', 'epoch': '0.005'}
100% 5/5 [02:28<00:00, 29.79s/it]
[*] Saving final LoRA adapters to ./grpo_cot_output/final_lora...
Unsloth: Restored added_tokens_decoder metadata in ./grpo_cot_output/final_lora/tokenizer_config.json.
[+] Training pipeline execution finished successfully!
