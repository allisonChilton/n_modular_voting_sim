# N-Modular Software Voting System Simulation
# Author: Allison Chilton <allison.chilton@colostate.edu>
# run and tested with Python 3.9

from termcolor import colored
from dataclasses import dataclass
import random
import math
import numpy as np
from typing import Tuple, Dict, Union, List
from enum import Enum
import pandas
import matplotlib.pyplot as plt

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

@dataclass
class FaultDetectionProbability:
    miss: float
    false_positive: float

@dataclass
class FaultDetectionRange:
    miss_range: Tuple[float, float]
    false_positive_range: Tuple[float, float]

    def random_configuration(self) -> FaultDetectionProbability:
        miss = random.uniform(self.miss_range[0], self.miss_range[1])
        false_positive = random.uniform(self.false_positive_range[0], self.false_positive_range[1])
        return FaultDetectionProbability(miss, false_positive)
    


class RandomSubsystemImplementation:
    def __init__(self, fault_categories: int, likelihood_range: FaultDetectionRange):
        aint = ord('A')
        assert 0 < fault_categories <= 26
        self.category_probs: Dict[str, FaultDetectionProbability] = {}
        for k in [chr(x) for x in range(aint,aint+fault_categories)]:
            self.category_probs[k] = likelihood_range.random_configuration()
    
    def vote(self, faults: Dict[str, bool], only_single_fault: bool = True) -> Dict[str, VoteResult]:
        retdict = {}
        single_fault_present = False # only allow one fault per vote
        for fault_type, is_present in faults.items():
            assert fault_type in self.category_probs, f"Unknown fault type category {fault_type}, subsystem only knows of {self.category_probs.keys()}"
            px = random.uniform(0.0,1.0) # randomly roll whether to be higher or lower than the probability out of 100%
            miss_prob = self.category_probs[fault_type].miss
            fp_prob = self.category_probs[fault_type].false_positive
            if is_present:
                hit_prob = (1 - miss_prob) + fp_prob
            else:
                hit_prob = fp_prob

            detect = px < hit_prob
            res = VoteResult.get_result(is_present, detect if not single_fault_present else is_present)
            if not res.is_correct():
                single_fault_present = only_single_fault
            
            retdict[fault_type] = res
        
        return retdict
            

class RandomSystemImplementation:
    def __init__(self, subsystems: int, fault_categories: int, miss_likelihood_ranges: List[FaultDetectionRange], alike: bool, weights = None):
        self.weights = weights if weights is not None else [1] * subsystems
        self.fault_categories = fault_categories
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


    def run_trial(self, fc: Dict[str, bool]) -> VoteResult:
        fault_present = True in fc.values()
        fault_detected = rsi.vote(fc)[0]
        result = VoteResult.get_result(fault_present, fault_detected)
        return result

    def collect_trials(self, trials: int, num_faults: int) -> pandas.DataFrame:
        results = []
        for i in range(trials):
            fc = get_fc(self.fault_categories, num_faults)
            res = self.run_trial(fc)
            rd = {'result': res.name, 'correct': res.is_correct(), 'detected': res.fault_detected, 'present': res.fault_present}
            results.append(rd)

        return pandas.DataFrame(results)


def get_fc(fault_categories: int, faults: int) -> Dict[str, bool]:
        aint = ord('A')
        assert 0 < fault_categories <= 26
        category_probs = {k: False for k in [chr(x) for x in range(aint,aint+fault_categories)]}
        while True:
            faultless_keys = [k for k,v in category_probs.items() if v == False]
            if fault_categories - len(faultless_keys) == faults or len(faultless_keys) == 0:
                break
            rk = random.choice(faultless_keys)
            category_probs[rk] = True
        return category_probs


if __name__ == "__main__":
    random.seed(42)
    rsi = RandomSystemImplementation(3, 5, [FaultDetectionRange((0.1, 0.3), (0.01, 0.05))] * 3, False)
    res = rsi.collect_trials(100, 1)
    print(res)
    #print(res.res_string())