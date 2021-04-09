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

    @staticmethod
    def get_result(is_present: bool, is_detected: bool) -> 'VoteResult':
        if not is_present and not is_detected:
            res = VoteResult.TRUE_NEGATIVE
        elif is_present and is_detected:
            res = VoteResult.TRUE_POSITIVE
        elif not is_present and is_detected:
            res = VoteResult.FALSE_POSITIVE
        elif is_present and not is_detected:
            res = VoteResult.FALSE_NEGATIVE
        return res
    
    @property
    def fault_present(self):
        return self == VoteResult.TRUE_POSITIVE or self == VoteResult.FALSE_NEGATIVE

    @property
    def fault_detected(self):
        return self == VoteResult.TRUE_POSITIVE or self == VoteResult.FALSE_POSITIVE

    def is_correct(self)->bool:
        return (self == VoteResult.TRUE_NEGATIVE or self == VoteResult.TRUE_POSITIVE)

    def res_string(self, colored_output=True):
        outstr = f"{i}: Fault {'' if self.fault_present else 'not '}present and {'' if self.fault_detected else 'not '}detected"
        if colored_output:
            outstr = colored(outstr, 'green' if self.is_correct() else 'red')
        return outstr

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
            px = random.uniform(0.0,1.0)
            pw = self.category_probs[fault_type]
            detect = px > (1 - pw if not is_present else pw)
            res = VoteResult.get_result(is_present, detect if not single_fault_present else is_present)
            if not res.is_correct():
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


    def run_trial(self, fc: Dict[str, bool]):
        fault_present = True in fc.values()
        fault_detected = rsi.vote(fc)[0]
        result = VoteResult.get_result(fault_present, fault_detected)
        return result


if __name__ == "__main__":
    random.seed(42)
    rsi = RandomSystemImplementation(3, 5, [(0.5, 0.5)] * 3, False)
    for i in range(100):
        fc = {'A': False, 'B': False, 'C' : True, 'D': False, 'E': False}
        res = rsi.run_trial(fc)
        print(res.res_string())