import os
from sympy.parsing.latex import parse_latex
import contextlib
import json
import random
from .utils import BaseReward

class MATHReward(BaseReward):
    def __init__(self, **kwargs):
        super(MATHReward, self).__init__(**kwargs)

    def CorrectnessReward(self, completions, prompts, target, do_log=True, do_update=True, **kwargs):
        """
        Evaluates completions based on the correctness of the final answer. 

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
        
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        log_entries = []  # Store log data
        for completion, prompt, gt in zip(completions, prompts, target):
            #extract the answer from the completion
            pred_answer = ExtractAnswerFromDataset(completion)
            #if the completion does not contain a boxed answer, return 0.0
            if pred_answer is None:
                reward = 0.0
            else:
                #check if the answer is correct
                are_equal = are_latex_expressions_equal(gt, pred_answer)
                if are_equal is None:
                    reward = 0.0
                else:
                    reward = float(are_equal)
            rewards.append(reward)

            # Log this completion
            if do_log:
                log_entries.append({
                    "prompt": prompt,
                    "completion": completion,
                    "target": gt,
                    "reward": reward
                })
        if do_log:
            # Save logs
            self.log_completions_to_file(log_entries)
        if do_update:
            self.update_datasets_with_ratios(kwargs, rewards)
        return rewards
        

###### utils #######

def ExtractAnswerFromDataset(text, pattern =  r'\boxed{' ):
    """Extracts all \\boxed{} contents from a LaTeX string, handling nested braces correctly."""
    results = []
    start = 0  

    while True:
        start = text.find(pattern, start)
        if start == -1:
            break  # No more \boxed{} found
        
        # Extract the content inside \boxed{}
        brace_count = 1
        content_start = start + len(pattern)
        i = content_start
        while i < len(text) and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            i += 1

        if brace_count == 0:
            #include the \boxed{} in the answer
            answer = text[start:i]
            results.append(answer)  # Extract content inside \boxed{}
            start = i  # Move past this \boxed{}
        else:
            break  # Unmatched braces (shouldn't happen in valid LaTeX)

    return results[-1] if results else None

def are_latex_expressions_equal(expr1, expr2):
    """Check if two LaTeX expressions are mathematically equivalent."""
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        try:
            expr1_ = parse_latex(expr1)
            expr2_ = parse_latex(expr2)
            equals = expr1_.equals(expr2_)
            if equals is None:
                return expr1 == expr2
            return equals
        except:
            return expr1 == expr2
    
def FormatRewardFunction(completions, **kwargs):
    """
    Only checking if the output format is correct, i.e.
    Format: thinking process... \\boxed{answer}
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion in completions:
        #extract the answer from the completion
        pred_answer = ExtractAnswerFromDataset(completion)
        #if the completion does not contain a boxed answer, return 0.0
        if pred_answer is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0)

    return rewards