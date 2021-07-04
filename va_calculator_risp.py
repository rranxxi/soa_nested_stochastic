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

def updateFundWeight(proj_fund_np, account_values, prev_fund_weight, t):
    F = proj_fund_np.shape[-1]
    for j in range(F):
        with np.errstate(divide='ignore'):
            prev_fund_weight[t,:,j] = proj_fund_np[t, :,j] / account_values[t]   
        mask = np.isnan(prev_fund_weight[t,:,j])  
        prev_fund_weight[t,:,j][mask] = 0       
        
class VACalculatorRisp:
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
        
     
    def projectionPeriod(self):
        """For 30 years, it will be 1 + 30 * valuationFrequency  = 121 
        Returns:
            [type]: [description]
        """
        return self.projection_period
    
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
    
    
    def benefitsCalculation(self, rw_id): 
        T = self.num_data_points # Number of projections 121
        I = self.inforce.numOfInforces() # Number of inforces 
        F = self.inforce.numOfFunds() # Number of funds 
        N = self.rn.numberOfScenarios() # Number of RN
        N = 10
        
        mapping = self.fund_mapping_matrix
        
        
        rw_k = self.rw.getScenario(rw_id)
        rw_k = np.exp(rw_k.dot(mapping))
         
        
        
        # Create output
        proj_fund_np = np.zeros([T, I, F]) # (T, 190000, 10)
        account_values =  np.zeros(proj_fund_np.shape[0: 2]) # (T, 190000)
        withdrawl_from_account_values = np.zeros_like(account_values) #(T, 190000)
        
        gmwb = np.zeros_like(account_values) # (T, 190000)
        gmwb_withdrawal = np.zeros_like(account_values) # (T, 190000)
        gmwb_balance = np.zeros_like(account_values) # (T, 190000)
        gmwb_payout = np.zeros_like(account_values) # (T, 190000)
        prev_fund_weight = np.zeros_like(proj_fund_np) # (T, 190000, 10)
        mbdb = np.zeros_like(account_values) # (T, 190000, 10)
        gmdb_payout = np.zeros_like(account_values) # (T, 190000)
        gmmb_payout = np.zeros_like(account_values) # (T, 190000)
        total_payout = np.zeros_like(account_values) # (T, 190000)
        pv_payout = np.zeros_like(account_values) # (T, 190000)

        # Load the inforce data
        if_fund_np = self.inforce.fundValuesNumpy()
        if_fund_fee_np_rate = self.inforce.fundFeeNumpy()
        if_gmwb_amount = self.inforce.inforcesGBAmount()
        if_gmwb_balance = self.inforce.inforcesGmwbBalance()    
        if_wd_rate = self.inforce.inforcesWithdrawalRate() / self.valuationFrequency()
        if_gmwb_withdrawal = self.inforce.inforcesGmwbWithdrawal()
        if_sex = self.inforce.inforcesGender()
        if_attained_age = self.inforce.inforcesAttainedAge()
        if_valid_proj_mat, if_valid_proj_attained_age_mat= self.inforce.getInforceValidProjectionMat() # (190000, 121)
        
        if_wd_phase_mask = if_gmwb_withdrawal > 0
        if_wd_rate_mask =  if_wd_rate > 0.0
    
        if_mortality_rate  = np.random.uniform(0, 0.001, [T, I])
        if_survivorship  = np.random.uniform(0, 0.001, [T, I])

        if_mortality_rate, if_survivorship = self.mortality_rate.getInforceMortalitySurvivor(if_attained_age, if_sex, self.projectionTermYears(), self.valuationFrequency())
        if_mortality_rate = if_mortality_rate.T
        if_survivorship = if_survivorship.T
        

        t = 0
        proj_fund_np[t] = if_fund_np
        gmwb[t] = if_gmwb_amount
        gmwb_withdrawal[t] = if_gmwb_withdrawal
        gmwb_balance[t] = if_gmwb_balance
        account_values[t] = np.sum(if_fund_np, axis = 1)
        updateFundWeight(proj_fund_np, account_values, prev_fund_weight, t)
            
        for t in np.arange(1, T):
            single_fund_return_ratio = rw_k[t]
            fund_growth = proj_fund_np[t-1] * single_fund_return_ratio
            
            fund_fee =  fund_growth * if_fund_fee_np_rate
            
            for j in range(F):
                fund_withdraw = prev_fund_weight[t - 1,:,j] * withdrawl_from_account_values[t - 1] 
                fund_growth[:,j] = np.maximum(0.0, fund_growth[:,j] - fund_withdraw - fund_fee[:,j])
            
            proj_fund_np[t] = fund_growth
            account_values[t] =  np.sum(proj_fund_np[t], axis = 1) 
            updateFundWeight(proj_fund_np, account_values, prev_fund_weight, t)
            # import pdb; pdb.set_trace()
            # Benefit Calculation
            gmwb[t][if_wd_phase_mask] = gmwb[t - 1][if_wd_phase_mask] - gmwb_withdrawal[t-1][if_wd_phase_mask]
            gmwb[t][~if_wd_phase_mask] = np.maximum(account_values[t][~if_wd_phase_mask], gmwb[t - 1][~if_wd_phase_mask])
            gmwb_withdrawal[t] = gmwb[t] * if_wd_rate
            
            gmwb_balance[t] = np.maximum(0, gmwb_balance[t - 1] - gmwb_withdrawal[t -1])
            gmwb_payout[t] = np.maximum(0, gmwb_withdrawal[t] - account_values[t]) * if_survivorship[t]
            withdrawl_from_account_values[t] = np.minimum(account_values[t], gmwb_withdrawal[t])
            
            mbdb[t] = gmwb[t]
            mbdb[t][if_wd_rate_mask]  = gmwb_balance[t][if_wd_rate_mask]
         
        
        gmdb_payout[1:] = np.maximum(0, mbdb[1:] - account_values[1:]) * if_mortality_rate[1:] * if_survivorship[1:]
        gmmb_payout = (mbdb - account_values) * if_survivorship
        total_payout = gmmb_payout * if_valid_proj_attained_age_mat.T + (gmdb_payout + gmwb_payout) * if_valid_proj_mat.T
        total_payout_sum = np.sum(total_payout, axis = 1)
        
        
        
        # Compute the first RW PV payout
        # Now Loop all the switch points
        # First compute No swtich with all RW first and cache the results
        # Compute backwards without polluting the cacehd RW results
        # For switch point K \in [T, 1], all the RW results from K to T will be polluted
        for sp in np.arange(T - 1, 1, -1, dtype=np.int32):
            # sp = 2
            # Lock the results from t = 0 to sp - 1
            for rn_id in range(1, N + 1): # Loop all the scenario
            # for rn_id in range(1, 2): # Loop all the scenario
                rn_l = self.rn.getScenario(rn_id)   
                rn_l = np.exp(rn_l.dot(self.fund_mapping_matrix))

                for t in np.arange(sp, T, dtype=np.int32):
                    single_fund_return_ratio = rn_l[t]
                    fund_growth = proj_fund_np[t-1] * single_fund_return_ratio
                    
                    fund_fee =  fund_growth * if_fund_fee_np_rate
                    
                    for j in range(F):
                        fund_withdraw = prev_fund_weight[t - 1,:,j] * withdrawl_from_account_values[t - 1] 
                        fund_growth[:,j] = np.maximum(0.0, fund_growth[:,j] - fund_withdraw - fund_fee[:,j])
                    
                    proj_fund_np[t] = fund_growth
                    account_values[t] =  np.sum(proj_fund_np[t], axis = 1) 
                    updateFundWeight(proj_fund_np, account_values, prev_fund_weight, t)
                    # import pdb; pdb.set_trace()
                    # Benefit Calculation
                    gmwb[t][if_wd_phase_mask] = gmwb[t - 1][if_wd_phase_mask] - gmwb_withdrawal[t-1][if_wd_phase_mask]
                    gmwb[t][~if_wd_phase_mask] = np.maximum(account_values[t][~if_wd_phase_mask], gmwb[t - 1][~if_wd_phase_mask])
                    gmwb_withdrawal[t] = gmwb[t] * if_wd_rate
                    
                    gmwb_balance[t] = np.maximum(0, gmwb_balance[t - 1] - gmwb_withdrawal[t -1])
                    gmwb_payout[t] = np.maximum(0, gmwb_withdrawal[t] - account_values[t]) * if_survivorship[t]
                    withdrawl_from_account_values[t] = np.minimum(account_values[t], gmwb_withdrawal[t])
                    
                    mbdb[t] = gmwb[t]
                    mbdb[t][if_wd_rate_mask]  = gmwb_balance[t][if_wd_rate_mask]
            
            gmwb *= if_valid_proj_mat.T
            mbdb *= if_valid_proj_mat.T
            gmdb_payout[1:] = np.maximum(0, mbdb[1:] - account_values[1:]) * if_mortality_rate[1:] * if_survivorship[1:]
            gmmb_payout = np.maximum(0, (mbdb - account_values)) * if_survivorship
            
            total_payout = gmmb_payout * if_valid_proj_attained_age_mat.T + (gmdb_payout + gmwb_payout) * if_valid_proj_mat.T
            total_payout_sum = np.sum(total_payout, axis = 1)
            import pdb; pdb.set_trace()
        
 
if __name__ == "__main__":
    vac = VACalculatorRisp(INFORCE_PATH, FUND_MAP_PATH, MORTALITY_PATH, ESG_PATH)
    vac.benefitsCalculation(1)