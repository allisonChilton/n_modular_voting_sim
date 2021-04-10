# N-Modular Software Voting System Simulation
# Author: Allison Chilton <allison.chilton@colostate.edu>
# run and tested with Python 3.9

from pandas.core.frame import DataFrame
from pathlib import Path
from termcolor import colored
from dataclasses import dataclass
import random
import math
import seaborn
import scipy.stats
import numpy as np
from typing import Tuple, Dict, Union, List
from enum import Enum
import pandas
import os
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
    def __init__(self, subsystems: int, fault_categories: int, likelihood_ranges: List[FaultDetectionRange], alike: bool, weights = None):
        self.weights = weights if weights is not None else [1] * subsystems
        self.fault_categories = fault_categories
        if not alike:
            self.subsystems = [RandomSubsystemImplementation(fault_categories, likelihood_ranges[i]) for i in range(subsystems)]
        else:
            self.subsystems = subsystems * [RandomSubsystemImplementation(fault_categories, likelihood_ranges[0])]
    
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
        fault_detected = self.vote(fc)[0]
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

class Scenarios:
    def __init__(self, trials, faults, fdr, fdr_low, subsystems, fault_categories):
        self.trials = trials
        self.faults = faults
        self.fdr = fdr
        self.fdr_low = fdr_low
        self.subsystems = subsystems
        self.fault_categories = fault_categories
    
    def _rsi_run(self, rsi_args, name):
        df = pandas.DataFrame()
        idx = 0
        while len(df) < self.trials:
            results = [df]
            rsi_obj = RandomSystemImplementation(*rsi_args)
            res = rsi_obj.collect_trials(20, self.faults)
            res['rsi_idx'] = idx
            results.append(res)
            df = pandas.concat(results)
            idx += 1
        
        df['trial_name'] = name
        return df


    def unweighted_disalike_subsystems(self):
        """Test a set of equally weighted disalike subsystems with a given subsystem count and fault category count"""
        rsi = (self.subsystems, self.fault_categories, [self.fdr] * self.subsystems, False)
        return self._rsi_run(rsi, "Unweighted_Disalike")

    def weighted_disalike_subsystems(self, good_weight: float):
        """Test a set of weighted disalike subsystems with a given subsystem count and fault category count,
        with a weighted probability favoring a system with better design tolerances, complemented by weaker systems with more lax tolerances"""
        other_weights = (1-good_weight) / (self.subsystems - 1)
        weights = [good_weight] + [other_weights] * (self.subsystems - 1)
        rsi = (self.subsystems, self.fault_categories, [self.fdr] * self.subsystems, False, weights)
        return self._rsi_run(rsi, "Weighted_Disalike")

    def unweighted_alike_subsystems(self):
        """Test a set of weighted disalike subsystems with a given subsystem count and fault category count,
        with a weighted probability favoring a system with better design tolerances, complemented by weaker systems with more lax tolerances"""
        rsi = (self.subsystems, self.fault_categories, [self.fdr], True)
        return self._rsi_run(rsi, "Unweighted_Alike")

    @staticmethod
    def run_suite(subsystems: int, fault_categories: int, trials: int, faults: int, fdr: FaultDetectionRange, fdr_low: FaultDetectionRange, good_weight: float):
        scn = Scenarios(trials, faults, fdr, fdr_low, subsystems, fault_categories)
        ud = scn.unweighted_disalike_subsystems()
        wd = scn.weighted_disalike_subsystems(good_weight)
        ua = scn.unweighted_alike_subsystems()
        total = pandas.concat([ud, wd, ua]).reset_index().rename({'index': 'trial_iteration'}, axis=1)
        return total

def ftest(p1, p2):
    v1 = np.var(p1)
    v2 = np.var(p2)
    f = v1 / v2
    dfn = p1.size - 1
    dfd = p2.size - 1
    p = 1-scipy.stats.f.cdf(f, dfn, dfd)
    return f, p


def analysis(results: DataFrame):
    df = results.drop(['trial_iteration', 'rsi_idx'], 1)
    uniq_names = df['trial_name'].nunique()
    tests_per_name = len(df) / uniq_names
    cor_df = df[df['correct'] == True].groupby('trial_name')['correct']
    pct_correct_by_type = cor_df.value_counts() / tests_per_name * 100
    std_dev_by_type = df.groupby('trial_name')['correct'].std()
    adf = pct_correct_by_type.to_frame().join(std_dev_by_type, on='trial_name', lsuffix="_percent", rsuffix="_stddev").reset_index().drop('correct', 1).set_index('trial_name')
    adf['p-value'] = pandas.NA

    # f-test
    unw_d = df[df['trial_name'] == "Unweighted_Disalike"]['correct']
    p1 = unw_d.to_numpy()
    for name in df['trial_name'].unique():
        if name == "Unweighted_Disalike":
            continue
        other_df = df[df['trial_name'] == name]['correct']
        p2 = other_df.to_numpy()
        f, p = ftest(p1, p2)
        adf.at[name, 'p-value'] = p


    return adf
    

smoke_test = False

def gen_table():
    scount = 3
    trials = 8000 if not smoke_test else 100
    cats = 5
    fdr =FaultDetectionRange((0.0, 0.3), (0.01, 0.05))
    fdrl = FaultDetectionRange((0.3, 0.5), (0.01, 0.05))
    gw = ((1 - 1/scount) + 0.05)
    one_fault = Scenarios.run_suite(
        subsystems=scount,
        fault_categories=cats,
        trials=trials,
        faults=1,
        fdr = fdr,
        fdr_low = fdrl,
        good_weight= gw
    )

    no_fault = Scenarios.run_suite(
        subsystems=scount,
        fault_categories=cats,
        trials=trials,
        faults=0,
        fdr = fdr,
        fdr_low = fdrl,
        good_weight= gw
    )

    comb_df = pandas.concat([one_fault, no_fault])
    adf = analysis(comb_df)

    # confusion matrix
    cm = pandas.DataFrame(comb_df['result'].value_counts().to_numpy().reshape(2,2), index=['TRUE','FALSE'], columns=['POSITIVE','NEGATIVE'])
    print(cm)
    hm = seaborn.heatmap(cm, annot=True, linewidths=0.2, fmt="5d", linecolor='gray').get_figure()
    hm.savefig("./images/cm.png")

    # prob dist
    stddevs = adf['correct_stddev'].to_numpy()
    pc = adf['correct_percent'].to_numpy()
    x = np.linspace(93,98,1000)
    x2 = np.linspace(-1, 1, 1000)
    pdfs = []
    pdfsn = []
    for avg, sd in zip(pc, stddevs):
        pdfs.append(scipy.stats.norm.pdf(x, loc=avg, scale=sd))
        pdfsn.append(scipy.stats.norm.pdf(x2, loc=0, scale=sd))
    
    fig, ax = plt.subplots(1, 1)
    for p, l in zip(pdfs, adf.index):
        ax.plot(x, p, label=l)
    ax.set_title("Probability Distribution of Different N-Module Configurations")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Percent Correct")
    ax.legend()
    fig.savefig('./images/probdist.png')

    fig, ax = plt.subplots(1, 1)
    for p, l in zip(pdfsn, adf.index):
        ax.plot(x2, p, label=l)
    ax.set_title("Probability Distribution of Different N-Module Configurations Normalized")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Normalized Percent Correct")
    ax.legend()
    fig.savefig('./images/probdistnorm.png')

    with open('fig1.tex', 'w') as latex_output:
        latex_output.write(r"\begin{figure}\caption{Results}")
        latex_output.write(adf.to_latex().replace("<NA>", "NA"))
        latex_output.write(f"Parameters: Trials = {trials}, Subsystems = {scount}, Fault Categories = {cats}, Good Weight = {gw:.2f}, Fault Miss Probability Range = {fdr.miss_range}")
        latex_output.write(f"False Positive Probability Range = {fdr.false_positive_range}, Worse Tolerance Miss and False Positive Probability Range = {(fdrl.miss_range, fdrl.false_positive_range)}\n")
        latex_output.write(r"\end{figure}")

def gen_plots(points=10):
    os.makedirs('images', exist_ok=True)
    scount = 3
    trials = 8000 if not smoke_test else 100
    cats = 5
    fdr =FaultDetectionRange((0.0, 0.3), (0.01, 0.05))
    fdrl = FaultDetectionRange((0.3, 0.5), (0.01, 0.05))
    gw = ((1 - 1/scount) + 0.05)
    dfs = []

    # vary fdr
    bmr = fdr.miss_range[1] * 0.5
    hmr = fdr.miss_range[1] * 1.5
    mra = np.linspace(bmr, hmr, points)
    subtrials = int(trials / points)
    for mr in mra:
        fdrn = FaultDetectionRange((0.0, mr), fdr.false_positive_range)
        one_fault = Scenarios.run_suite(
            subsystems=scount,
            fault_categories=cats,
            trials=subtrials,
            faults=1,
            fdr = fdrn,
            fdr_low = fdrl,
            good_weight= gw
        )

        no_fault = Scenarios.run_suite(
            subsystems=scount,
            fault_categories=cats,
            trials=subtrials,
            faults=0,
            fdr = fdrn,
            fdr_low = fdrl,
            good_weight= gw
        )

        dfs.append(analysis(pandas.concat([one_fault, no_fault])))

    correct_data = []
    std_devs = []
    ps = []
    for mr, df in zip(mra, dfs):
        cpdf = df['correct_percent'].to_frame().rename({'correct_percent': mr}, axis=1)
        sddf = df['correct_stddev'].to_frame().rename({'correct_stddev': mr}, axis=1)
        pdf = df['p-value'].to_frame().rename({'p-value': mr}, axis=1)
        correct_data.append(cpdf)
        std_devs.append(sddf)
        ps.append(pdf)
    
    fig_paths = ['./images/percent_correct_fdr.png', './images/stddev_fdr.png', './images/pvalue_fdr.png']
    cpdf = pandas.concat(correct_data, axis=1).T
    ax = cpdf.plot()
    ax.set_title("Percent correct by varying fault miss probability")
    ax.set_xlabel("Maximum Fault Miss Probability")
    ax.set_ylabel("Percent correct")
    fig = ax.figure
    fig.savefig(fig_paths[0])

    sddf = pandas.concat(std_devs, axis=1).T
    ax = sddf.plot()
    ax.set_title("Std Dev by varying fault miss probability")
    ax.set_xlabel("Maximum Fault Miss Probability")
    ax.set_ylabel("Standard Deviation")
    fig = ax.figure
    fig.savefig(fig_paths[1])

    pdf = pandas.concat(ps, axis=1).T.drop('Unweighted_Disalike', 1)
    ax = pdf.plot()
    ax.set_title("P-Value (compared to unweighted disalike) by varying fault miss probability")
    ax.set_xlabel("Maximum Fault Miss Probability")
    ax.set_ylabel("P-Value")
    fig = ax.figure
    fig.savefig(fig_paths[2])

    with open('plots.tex', 'w') as f:
        for figp in [Path(x) for x in fig_paths]:
            f.write(r"\begin{figure}\includegraphics[width=8cm]{")
            f.write(figp.with_suffix("").name)
            f.write(r"}\centering\end{figure}")
        
        #confusion matrix
        f.write(r"\begin{figure}\includegraphics[width=8cm]{cm}\centering\end{figure}")

        # pdf
        f.write(r"\begin{figure}\includegraphics[width=8cm]{probdist}\centering\end{figure}")

        # pdfn
        f.write(r"\begin{figure}\includegraphics[width=8cm]{probdistnorm}\centering\end{figure}")


if __name__ == "__main__":
    random.seed(42)
    gen_table()
    gen_plots()