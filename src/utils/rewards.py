
def ParityReward(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth_list = ground_truth.split()
    length = len(ground_truth_list)

    try:
        solution_list = solution_str.split()[:length]
    except:
        return 0
    
    n_equal = 0
    for i in range(0, length, 2):
        yi = int(ground_truth_list[i])
        zi = int(ground_truth_list[i+1])
        try:
            hat_yi = int(solution_list[i])
            hat_zi = int(solution_list[i+1])
        except:
            # If the solution is not in the expected format, return 0
            return 0
        if yi ^ zi != hat_yi ^ hat_zi:
            # If the parity does not match, return 0
            n_equal += 1
    
    # the model gets 0.1 reward if the length of the solution is equal to the length of the ground truth and 
    # all tokens are integers (hopefully they are 0 or 1)
    reward = 0.1 * (length==len(solution_str.split())) + 0.9 * (n_equal == 0)

    return reward

