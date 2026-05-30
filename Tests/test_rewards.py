import sys
import os
# Add src directory to path so we can import rewards
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "phase4")))

from rewards import (
    extract_xml_answer,
    format_reward_fn,
    math_correctness_reward_fn,
    p_grpo_format_reward_fn,
    step_grpo_reward_fn,
    tag_spam_penalty_fn
)

def test_extract_xml_answer():
    # Test typical extraction with think tags
    assert extract_xml_answer("<think>I will add 2 and 2</think> The answer is 4") == "4"
    
    # Test \boxed{...} extraction
    assert extract_xml_answer("The result is \\boxed{24}") == "24"
    assert extract_xml_answer("<think>solve</think> \\boxed{-5.5}") == "-5.5"
    
    # Test last number extraction when multiple are present
    assert extract_xml_answer("First we get 10, then we get 20") == "20"
    
    # Test comma removal
    assert extract_xml_answer("The total is 1,234,567") == "1234567"
    
    # Test no numbers
    assert extract_xml_answer("There are no digits here") == ""

def test_format_reward_fn():
    prompts = [""] * 6
    completions = [
        "<think>Let's calculate</think> The answer is 4",       # Perfect (1.0)
        "<think>Incomplete tags",                                # Unclosed (0.2)
        "No tags whatsoever 4",                                  # None (0.0)
        "<think>Empty</think> ",                                 # No content after </think> (0.5)
        "<think>A</think><think>B</think> The answer is 4",       # Multi-block format penalty (0.3)
        "<think>A</think> <think>B</think> </think> The answer is 4" # Multi-block tag mismatch (0.3)
    ]
    rewards = format_reward_fn(prompts, completions)
    assert rewards[0] == 1.0
    assert rewards[1] == 0.2
    assert rewards[2] == 0.0
    assert rewards[3] == 0.5
    assert rewards[4] == 0.3
    assert rewards[5] == 0.3

def test_math_correctness_reward_fn():
    prompts = [""] * 3
    completions = [
        "The answer is 12",
        "The answer is 12",
        "The answer is 13"
    ]
    targets = ["12", "12 ", "12"]
    rewards = math_correctness_reward_fn(prompts, completions, targets)
    assert rewards[0] == 1.0
    assert rewards[1] == 1.0 # strips target whitespace
    assert rewards[2] == 0.0

    # Test float/decimal equivalence (e.g. 16.00 vs 16)
    rewards_float = math_correctness_reward_fn(
        [""] * 3,
        ["The answer is 16.00", "The answer is 16", "The answer is 16.05"],
        ["16", "16.0", "16"]
    )
    assert rewards_float[0] == 1.0
    assert rewards_float[1] == 1.0
    assert rewards_float[2] == 0.0


def test_p_grpo_format_reward_fn():
    prompts = [""] * 5
    completions = [
        "<think>thinking</think> 12",                     # Correct answer + correct format (1.0)
        "<think>thinking</think> 13",                     # Incorrect answer + correct format (0.0)
        "No format 12",                                   # Correct answer + no format (0.2)
        "<think>A</think><think>B</think> 12",            # Correct answer + multi-block formatting penalty (0.3)
        "<think>unclosed 12"                              # Correct answer + unclosed block penalty (0.2)
    ]
    targets = ["12", "12", "12", "12", "12"]
    rewards = p_grpo_format_reward_fn(prompts, completions, targets)
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0 # zeroed out because incorrect
    assert rewards[2] == 0.2
    assert rewards[3] == 0.3
    assert rewards[4] == 0.2

def test_step_grpo_reward_fn():
    prompts = [""] * 6
    completions = [
        "<think>solve it</think> 12",                                            # Correct, 0 steps -> 1.0
        "<think>wait, let's check. hmm, this is 12</think> 12",                   # Correct, 2 steps ('wait', 'hmm') -> 0.99^2 = 0.9801
        "<think>wait, let's check. hmm, this is 12</think> 13",                   # Incorrect -> 0.0
        "wait, let's check. hmm, this is 12",                                     # Correct, 2 steps (no tags) -> 0.9801
        "<think>wait</think><think>hmm</think> 12",                               # Multi-block exploit test: captures both 'wait' and 'hmm' (2 steps) and applies block penalty: 0.99^2 - 0.5 = 0.4801
        "<think>wait, actually this is 12"                                        # Unclosed block test: fallback extracts after <think>, 2 steps ('wait', 'actually') -> 0.99^2 = 0.9801
    ]
    targets = ["12", "12", "12", "12", "12", "12"]
    rewards = step_grpo_reward_fn(prompts, completions, targets)
    
    assert rewards[0] == 1.0
    assert abs(rewards[1] - 0.9801) < 1e-6
    assert rewards[2] == 0.0
    assert abs(rewards[3] - 0.9801) < 1e-6
    # Multi-block penalty: 0.99^2 - 0.5 = 0.4801
    assert abs(rewards[4] - 0.4801) < 1e-6
    # Unclosed fallback: 0.99^2 = 0.9801
    assert abs(rewards[5] - 0.9801) < 1e-6

def test_tag_spam_penalty_fn():
    """Verify the electrified fence catches all tag-spam variants."""
    prompts = [""] * 7
    completions = [
        # 1. Clean completion with only <think> — no penalty
        "<think>Let me calculate step by step.\n16 - 3 = 13\n13 - 4 = 9\n9 * 2 = 18</think>\nThe answer is 18.",
        # 2. HTML tag spam: <p>, <br>, <strong> — 3 unique types -> -0.9
        "<think>Let me calculate\n<p>Eggs: 16</p>\n<br>\n<strong>Result: 18</strong></think>\n18",
        # 3. Digit tags: <1>, <2>, <3> — 1 unique type (__digit__) -> -0.3
        "<think>Step <1> get eggs\nStep <2> subtract\nStep <3> multiply</think>\n18",
        # 4. Invented tags: <calculate>, <result> — 2 unique types -> -0.6
        "<think><calculate>16 - 3 = 13</calculate>\n<result>13 - 4 = 9</result></think>\n18",
        # 5. Mixed HTML + digit + invented — 4 unique types -> -1.2
        "<think><li>Eggs: 16</li>\n<b>Bold text</b>\n<1>Subtract 3</1>\n<answer>18</answer></think>\n18",
        # 6. Mega spam — 6+ unique types should cap at -1.5
        "<think><p>A</p><br><div><strong><ul><li>B</li></ul></strong></div></think>\n18",
        # 7. No tags at all — no penalty
        "Just plain text reasoning. 16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18.",
    ]
    penalties = tag_spam_penalty_fn(prompts, completions)
    
    # 1. Clean <think> only -> 0.0
    assert penalties[0] == 0.0, f"Clean completion should get 0.0 penalty, got {penalties[0]}"
    # 2. HTML spam (p, br, strong) -> -0.9
    assert abs(penalties[1] - (-0.9)) < 1e-6, f"HTML spam should get -0.9, got {penalties[1]}"
    # 3. Digit tags -> -0.3
    assert abs(penalties[2] - (-0.3)) < 1e-6, f"Digit tags should get -0.3, got {penalties[2]}"
    # 4. Invented tags (calculate, result) -> -0.6
    assert abs(penalties[3] - (-0.6)) < 1e-6, f"Invented tags should get -0.6, got {penalties[3]}"
    # 5. Mixed types -> -1.2
    assert abs(penalties[4] - (-1.2)) < 1e-6, f"Mixed spam should get -1.2, got {penalties[4]}"
    # 6. Mega spam -> capped at -1.5
    assert abs(penalties[5] - (-1.5)) < 1e-6, f"Mega spam should cap at -1.5, got {penalties[5]}"
    # 7. No tags at all -> 0.0
    assert penalties[6] == 0.0, f"No tags should get 0.0 penalty, got {penalties[6]}"

if __name__ == "__main__":
    print("[*] Running unit tests...")
    test_extract_xml_answer()
    test_format_reward_fn()
    test_math_correctness_reward_fn()
    test_p_grpo_format_reward_fn()
    test_step_grpo_reward_fn()
    test_tag_spam_penalty_fn()
    print("[+] All reward tests passed successfully!")
