import numpy as np
import cupy as cnp

import pandas as pd
import  math, pdb, tqdm
from math import ceil
from datetime import datetime
from copy import deepcopy
import time

from sos_utils import get_fund_mapping_matrix, fund_remapping 
from sos_utils import random_select_inforces, merge_rw_rn_scenario_full

from scenarios import ScenarioLoader
from inforce   import Inforce
from mortality import Mortality
from benefits import gmdbCalculation, get_benefits_ground_truth_from_excel

from params import *


class VACalculator:
    def __init__(self, inforce_path, fund_mapping_path, mortality_path, soa_esg_scenario_path, valuation_frequency = 4,  projection_term_years = 30):
        self.valuation_frequency = valuation_frequency
        
        self.projection_term_years = projection_term_years
        # Load fund to index mapping
        self.fund_mapping_matrix = get_fund_mapping_matrix(fund_mapping_path)
        
        self.inforce = Inforce(inforce_path, self.fund_mapping_matrix)
        # Load RW scenario
        self.rw = ScenarioLoader(soa_esg_scenario_path, start_idx = 0, max_num_scenarios = 500)
        # Load RN scenarios
        self.rn = ScenarioLoader(soa_esg_scenario_path, start_idx = 500, max_num_scenarios = 500)

        self.risk_free_interest_rate = CFG_RISK_FREE_IR
        self.mortality_rate = Mortality(mortality_path)    
        self.num_data_points = self.valuation_frequency * self.projection_term_years + 1 
        
        self.projection_period = np.linspace(0, self.num_data_points - 1, self.num_data_points, dtype=np.int32)
        self.discount_factor = self.computeAccumDiscountFactor()

    def valuationFrequency(self):
        """The evaluation steps for each year

        Returns:
            int: how many projection points per year
        """
        return self.valuation_frequency

    def valuationDurationMonths(self):
        """The evaluation steps for each year

        Returns:
            int: how many projection points per year
        """
        return 12 / self.valuation_frequency
    
    def projectionPeriod(self):
        """For 30 years, it will be 1 + 30 * valuationFrequency  = 121 
        Returns:
            [type]: [description]
        """
        return self.projection_period
    

    def riskFreeInterestRate(self):
        """Risk-free interest rate

        Returns:
            float: A constane value of risk free interest rate. Loaded from assumptions
        """
        return self.risk_free_interest_rate

    def projectionTermYears(self):
        """Number of projection years
        Returns:
            int: a constant value of projection years: 30 
        """
        return self.projection_term_years
    
    def computeAccumDiscountFactor(self):
        dist_fact = np.exp(0.0 - self.riskFreeInterestRate() * (1.0 / self.valuationFrequency()) * self.projectionPeriod())
        return dist_fact
    
    def getDiscountFactor(self, rn_start_time = 0):
        dis_factor = np.zeros_like(self.discount_factor)
        dis_factor_len = len(dis_factor[rn_start_time:])
        dis_factor[rn_start_time:] = self.discount_factor[0 : 0 + dis_factor_len]
        return dis_factor
    
    
    def VAPayoutCalculation(self, rw_rn_merged_scenarios, inforce_id):        
        try:
            # Get the original inforce fund fee 10 x 1
            if_fund_fee_np_rate = self.inforce.fundFeeNumpy()
            # Get the original inforce fund balance 10 x 1
            if_fund_np = self.inforce.fundValuesNumpy()[inforce_id]
            if_gmwb_amount = self.inforce.inforcesGBAmount()[inforce_id]
            if_gmwb_balance = self.inforce.inforcesGmwbBalance()[inforce_id]
            if_wd_rate = self.inforce.inforcesWithdrawalRate()[inforce_id] / self.valuationFrequency()
            if_gmwb_withdrawal = self.inforce.inforcesGmwbWithdrawal()[inforce_id]
            if_sex = self.inforce.inforcesGender()[inforce_id]
            if_age = int(self.inforce.inforcesAttainedAge()[inforce_id])
            if_mortality_rate, if_survivorship = self.mortality_rate.getMortalityRateFromAge(if_age, self.projection_term_years, is_male = if_sex == 'M')
            if_proj_data_points = ceil(self.inforce.getMaxProjectionMonthsRecordId(inforce_id) * 1.0 / self.valuationDurationMonths())
        except:
            import pdb; pdb.set_trace()
            
        if_proj_data_points = min(if_proj_data_points, rw_rn_merged_scenarios.shape[1])
        
        if_fund_fee_np_rate = if_fund_fee_np_rate.ravel()
        # import pdb; pdb.set_trace()
        proj_fund_np = np.zeros_like(rw_rn_merged_scenarios) # 120 x 121 X 10
        account_values =  np.zeros(rw_rn_merged_scenarios.shape[0:2]) # 120 x 121 X 1
        withdrawl_from_account_values = np.zeros_like(account_values) # 120 x 121 X 1
        
        gmwb = np.zeros_like(account_values) # 120 x 121 X 1
        gmwb_withdrawal = np.zeros_like(account_values) # 120 x 121 X 1
        gmwb_balance = np.zeros_like(account_values) # 120 x 121 X 1
        gmwb_payout = np.zeros_like(account_values) # 120 x 121 X 1
        prev_fund_weight = np.zeros_like(if_fund_np) # 10 x 1
        
        # import pdb; pdb.set_trace()
        # 120 x 121 x 10
        for i in range(account_values.shape[0]): # 0 -> 119. 1 RW, 1RN -> 121 x 10
            for t in range(if_proj_data_points + 1):  #0 -> 120
                within_withdrawl_phase = if_gmwb_withdrawal > 0
                if t == 0:
                    proj_fund_np[i, t] = if_fund_np
                    withdrawl_from_account_values[i, t] = 0
                    
                    gmwb[i, 0] = if_gmwb_amount
                    gmwb_withdrawal[i, 0] = if_gmwb_withdrawal
                    
                    gmwb_balance[i, 0] = if_gmwb_balance
                    gmwb_payout[i, t] = 0
                    

                    sum_av = np.sum(if_fund_np)
                    account_values[i, t] = sum_av
                                        
                    for j in range(0, len(if_fund_np)):
                        prev_fund_weight[j] = if_fund_np[j] / sum_av
                else:
                    fund_growth = proj_fund_np[i, t - 1]  * rw_rn_merged_scenarios[i, t]
                    
                    # Compute fund withdrawl for each fund 10 x 1
                    # t1: must be zero 
                    fund_withdraw = prev_fund_weight * withdrawl_from_account_values[i, t - 1]
                    
                    # import pdb; pdb.set_trace()
                    fund_fee =  fund_growth * if_fund_fee_np_rate
                    
                    fund_fee_plus_withdraw = fund_fee + fund_withdraw
                    
                    fund_remain = fund_growth * (1.0 - if_fund_fee_np_rate) - fund_withdraw
                    
                    proj_fund_np[i, t] = np.maximum(0.0, fund_remain)
                    
                    # Compute beneits
                    # GMWB
                    account_values[i, t] = np.sum(proj_fund_np[i, t]) 

                    
                    for j in range(0, len(proj_fund_np[i, t])):
                        prev_fund_weight[j] = proj_fund_np[i, t, j] / account_values[i, t]

                        
                    if within_withdrawl_phase:
                        gmwb[i, t] = gmwb[i, t-1] - gmwb_withdrawal[i, t - 1]
                    else:
                        gmwb[i, t] = max(account_values[i, t], gmwb[i, t-1])
                    
                    # GMWB_withdrawal
                    gmwb_withdrawal[i, t] = gmwb[i, t] * if_wd_rate
                    
                    # GMWB_withdrawal balance will be used for the gmdb computation 
                    gmwb_balance[i, t] = max(0, gmwb_balance[i, t-1] - gmwb_withdrawal[i, t-1])
                    gmwb_payout[i, t] = max(0, gmwb_withdrawal[i, t] - account_values[i, t]) * if_survivorship[t]
                    withdrawl_from_account_values[i, t] = min(account_values[i, t], gmwb_withdrawal[i, t])
                    
        mbdb = gmwb_balance if if_wd_rate > 0.0 else gmwb
        mbdb, gmdb_payout, gmmb_payout = gmdbCalculation(gmwb, gmwb_balance, mbdb, if_mortality_rate, if_survivorship, account_values, if_proj_data_points)
        total_payout = gmmb_payout + gmdb_payout + gmwb_payout

        # Apply the discount factor to copmpute the pv payout
        pv_payout = np.zeros_like(total_payout)
        # import pdb; pdb.set_trace()
        for i in range(total_payout.shape[0]):
            rn_start_time = i + 1
            dist_factor = self.getDiscountFactor(rn_start_time)
            pv_payout[i] = total_payout[i] * dist_factor
        return pv_payout
        
 
if __name__ == "__main__":
    vac = VACalculator(INFORCE_PATH, FUND_MAP_PATH, MORTALITY_PATH, ESG_PATH)

    max_real_word_scenarios = vac.rw.numberOfScenarios()
    max_risk_neutral_scenarios = vac.rn.numberOfScenarios()
    
    num_inforces = 100
    inforce_list = random_select_inforces(num_inforces, 0, 10000)
    
    max_real_word_scenarios = 10
    max_risk_neutral_scenarios = 200

    num_rw = max_real_word_scenarios
    num_rn = max_risk_neutral_scenarios
    
    
    rw_pv_payout= []
    start = time.time()
    for rw_id in range(0, max_real_word_scenarios):
        rn_pv_payout = np.zeros([max_risk_neutral_scenarios, 121])
        
        for rn_id in range(0, max_risk_neutral_scenarios):
            print(rw_id, rn_id)
            rw_i = vac.rw.getScenario(rw_id)
            rn_j = vac.rn.getScenario(rn_id)
            merged_rw_rn = merge_rw_rn_scenario_full(rw_i, rn_j)
            merged_scenarios_return = np.exp(merged_rw_rn.dot(vac.fund_mapping_matrix))
            
            # loop the inforce to get sum of the RN present payout value of inforces
            for if_id in range(len(inforce_list)):
                inforce_id = inforce_list[if_id]
                ret = vac.VAPayoutCalculation(merged_scenarios_return, inforce_id)
                rn_pv_payout[rn_id] += np.sum(ret, axis = 0)
                            
        dist_factor_t0 = vac.getDiscountFactor(1)
        rw_payout_mean = np.mean(rn_pv_payout, axis = 0)        
        rw_pv_payout.append(np.sum(dist_factor_t0 * rw_payout_mean))  
        # import pdb; pdb.set_trace()    
    end = time.time()
    print('delay for ', num_rw, num_rn, num_inforces, end - start, ' secs')
    rw_pv_payout = np.sort(np.array(rw_pv_payout))
    import pdb; pdb.set_trace()
    
    
     