"""
Basic statistics of data
"""

from collections import Counter


def data_statistics(input_list):
    # Write your code here.

    # Mean
    mean = sum(input_list) / len(input_list)

    # Median
    if len(input_list) % 2 == 1:
        median = sorted(input_list)[len(input_list)//2]
    else:
        median = (sorted(input_list)[len(input_list)//2] + sorted(input_list)[(len(input_list)//2)-1]) / 2
    
    # Mode
    mode_dict = Counter(input_list)
    mode, _ = mode_dict.most_common(1)[0]
    
    # Standard Deviation
    std = ((sum([(i-mean)**2 for i in input_list]))/(len(input_list) - 1))**(1/2)
    
    # Variance
    var = std**2

    # Confidence Interval
    c = 1.96
    temp = c*(std/(len(input_list)**0.5))

    # print(mode.most_common)
    
    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "sample_variance": var,
        "sample_standard_deviation": std,
        "mean_confidence_interval": [mean-temp, mean+temp],
    }
