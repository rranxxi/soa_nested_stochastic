import numpy as np
import pandas as pd
        
def gmdbCalculation(gmwb, gmwbBalance, mbdb, mortality_rate, survivorship, account_values, last_projection_index):    
    gmdb_payout = np.zeros_like(account_values)
    gmmb_payout = np.zeros_like(account_values)
    for i in range(account_values.shape[0]):
        gmdb_payout[i, 1:] = np.maximum(0, mbdb[i, 1:] - account_values[i, 1:]) * mortality_rate[1:] * survivorship[1:]
        # Only the last year has the value
        gmmb_payout[i, last_projection_index] = mbdb[i, last_projection_index]   - account_values[i, last_projection_index] 
        gmmb_payout[i, last_projection_index] *= survivorship[last_projection_index]
        
    return mbdb, gmdb_payout, gmmb_payout


def get_benefits_ground_truth_from_excel(excel_path, sheet_name):        
    df = pd.read_excel(excel_path, sheet_name = sheet_name)
    d = {}
    # d['total_fund_value_before_fee_and_withdrawl_gt'] = df['Total Fund Value (before fees and withdrawal)']
    # d['total_fees'] = df['Total Fees']
    # d['withdrawal_from_av'] = df['Withdrawal from AV']
    d['gmwb_gt'] = df['GMWB']
    d['gmwb_balance_gt'] = df['GMWB Balance']
    d['gmwb_withdrawal_gt'] = df['GMWB Withdrawal']
    d['gmwb_payout_gt'] = df['GMWB Payout']
    d['gmdbmb_gt'] = df['GMDB/GMMB']
    d['pv_payout'] = df['PV Payout']
    d['df_gt'] = df
    return d    
    
        