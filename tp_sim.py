from termcolor import colored
import random
import math
import numpy as np
from typing import Tuple, Dict, Union, List
from enum import Enum

class VoteResult(Enum):
    TRUE_NEGATIVE = 1
    TRUE_POSITIVE = 2
    FALSE_NEGATIVE = 3
    FALSE_POSITIVE = 4

    def __bool__(self):
        return (self == VoteResult.TRUE_POSITIVE or self == VoteResult.FALSE_POSITIVE)

class RandomSubsystemImplementation:
    def __init__(self, fault_categories: int, miss_likelihood_range: Tuple[float, float]):
        aint = ord('A')
        assert 0 < fault_categories <= 26
        self.category_probs = {k: random.uniform(miss_likelihood_range[0], miss_likelihood_range[1]) for k in [chr(x) for x in range(aint,aint+fault_categories)]}
    
    def vote(self, faults: Dict[str, bool], only_single_fault: bool = True) -> Dict[str, VoteResult]:
        retdict = {}
        single_fault_present = False # only allow one fault per vote
        for fault_type, is_present in faults.items():
            assert fault_type in self.category_probs, f"Unknown fault type category {fault_type}, subsystem only knows of {self.category_probs.keys()}"
            detect = random.uniform(0.0,1.0) > self.category_probs[fault_type]
            if not is_present and (not detect or single_fault_present):
                res = VoteResult.TRUE_NEGATIVE
            elif is_present and (detect or single_fault_present):
                res = VoteResult.TRUE_POSITIVE
            elif not is_present and detect:
                res = VoteResult.FALSE_POSITIVE
                single_fault_present = only_single_fault
            elif is_present and not detect:
                res = VoteResult.FALSE_NEGATIVE
                single_fault_present = only_single_fault
            
            retdict[fault_type] = res
        
        return retdict
            

class RandomSystemImplementation:
    def __init__(self, subsystems: int, fault_categories: int, miss_likelihood_ranges: List[Tuple[float, float]], alike: bool, weights = None):
        self.weights = weights if weights is not None else [1] * subsystems
        if not alike:
            self.subsystems = [RandomSubsystemImplementation(fault_categories, miss_likelihood_ranges[i]) for i in range(subsystems)]
        else:
            self.subsystems = subsystems * [RandomSubsystemImplementation(fault_categories, miss_likelihood_ranges[0])]
    
    def vote(self, faults: Dict[str, bool], only_single_fault: bool = True) -> Tuple[bool, List[Dict[str, VoteResult]]]:
        votes = [subsystem.vote(faults, only_single_fault) for subsystem in self.subsystems]

        true_votes = 0
        for subsys_vote in votes:
            for fault_category_vote in subsys_vote.values():
                if bool(fault_category_vote):
                    true_votes += 1
                    break
        majority_vote = true_votes > (len(self.subsystems) / 2)

        return majority_vote, votes


if __name__ == "__main__":
    random.seed(42)
    rsi = RandomSystemImplementation(3, 5, [(0.4, 0.5)] * 3, False)
    for i in range(100):
        fc = {'A': True, 'B': False, 'C' : False, 'D': False, 'E': False}
        fault_present = True in fc.values()
        fault_detected = rsi.vote(fc)[0]
        correct = (fault_present and fault_detected) or (not fault_present and not fault_detected)
        print(colored(f"{i}: Fault {'' if fault_present else 'not '}present and {'' if fault_detected else 'not '}detected", 'green' if correct else 'red'))