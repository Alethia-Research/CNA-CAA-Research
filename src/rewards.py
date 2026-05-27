import re

def get_completion_text(comp) -> str:
    """
    Safely extracts the generated text string from a completion.
    If the completion is a list of message dicts (from a chat template),
    it retrieves the content of the assistant's message.
    """
    if isinstance(comp, list):
        for msg in comp:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # Fallback to the last message's content
        if len(comp) > 0 and isinstance(comp[-1], dict):
            return comp[-1].get("content", "")
    elif isinstance(comp, str):
        return comp
    return ""

def extract_xml_answer(text: str) -> str:
    """
    Extracts the final answer from the model completion.
    Looks after the </think> tag if present, and extracts either:
    1. A number wrapped in \boxed{...}
    2. The last number found in the text.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    # Remove commas from numbers to avoid matching errors (e.g., 1,000 -> 1000)
    text_clean = text.replace(",", "")
    
    # 1. Look for \boxed{...}
    boxed_match = re.search(r"\\boxed\{([0-9.-]+)\}", text_clean)
    if boxed_match:
        return boxed_match.group(1)
        
    # 2. Look for the last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_clean)
    if numbers:
        return numbers[-1]
        
    return ""

def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """
    Standard formatting reward. Checks if the model output follows the <think>...</think> structure.
    Returns:
        1.0 for perfect structure.
        0.5 for partial/malformed structure.
        0.0 for no structure.
    """
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        
        # Check start and end tags
        has_start = "<think>" in comp_text
        has_end = "</think>" in comp_text
        
        if has_start and has_end:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            # Ensure correct ordering and some content after </think>
            if start_idx < end_idx and len(comp_text[end_idx + 8:].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif has_start or has_end:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards

def math_correctness_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Standard math correctness reward.
    Compares the extracted numerical answer with the target ground truth.
    """
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        
        if extracted and extracted == target_clean:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def p_grpo_format_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Posterior-GRPO Formatting Reward.
    Zeroes out the formatting (process) reward if the final math answer is incorrect.
    This prevents the model from being reinforced for 'looking smart' while getting the math wrong.
    """
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        
        # Check correctness first
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = (extracted and extracted == target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        # If correct, evaluate format
        has_start = "<think>" in comp_text
        has_end = "</think>" in comp_text
        if has_start and has_end:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8:].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        else:
            rewards.append(0.2)
            
    return rewards

def step_grpo_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Step-GRPO Decaying Reward.
    If the answer is correct, applies an exponential decay based on the number
    of cognitive transition steps (tokens like 'Wait', 'Hmm') inside the monologue.
    R_j = gamma^j * 𝟙[correct]
    """
    rewards = []
    gamma = 0.99
    # Transition tokens indicating a new step/reasoning transition
    transition_tokens = ["wait", "hmm", "but", "thinking", "actually", "let me check"]
    
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = (extracted and extracted == target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        # Extract the monologue inside <think>...</think>
        think_content = ""
        if "<think>" in comp_text and "</think>" in comp_text:
            think_content = comp_text.split("<think>")[-1].split("</think>")[0].lower()
        else:
            think_content = comp_text.lower()
            
        # Count occurrence of cognitive transition tokens
        steps = 0
        for token in transition_tokens:
            steps += think_content.count(token)
            
        # Calculate decayed reward
        decayed_reward = float(gamma ** steps)
        rewards.append(decayed_reward)
        
    return rewards
