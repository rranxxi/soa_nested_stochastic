import numpy as np
import pandas as pd

class Inforce:
    def __init__(self, inforce_path, fund_mapping_matrix):
        self.df = pd.read_csv(inforce_path)
        # Remove the whitespaces in the names
        self.df = self.df.rename(columns=lambda x: x.strip())
        self.df.columns = self.df.columns.str.replace(' ', '')

        self.fund_mapping_matrix = fund_mapping_matrix
        self.num_of_funds = 10 
        # Mapped index df     
        self.df_index = self.calculateMappedIndex()   
        

    def numOfFunds(self):
        return self.num_of_funds
    
    def numOfInforces(self):
        return self.df.shape[0]
    
    def inforcesRollUpRate(self):
        return self.df['rollUpRate'].to_numpy()

    def inforcesGBAmount(self):
        return self.df['gbAmt'].to_numpy()

    def inforcesGmwbBalance(self):
        return self.df['gmwbBalance'].to_numpy()

    def inforcesWithdrawalRate(self):
        return self.df['wbWithdrawalRate'].to_numpy()
        
    def inforcesGmwbWithdrawal(self):
        return self.df['withdrawal'].to_numpy()

    def inforcesGender(self):
        return self.df['gender'].to_numpy()

    def inforcesAttainedAge(self):
        return self.df['AttainedAge'].to_numpy().astype(np.int32)
    
    def fundValuesNumpy(self):
        # fund1_index = list(self.df.columns).index('FundValue1')
        fund1_index = self.df.columns.get_loc("FundValue1")
        return self.df.iloc[:,fund1_index : fund1_index + self.num_of_funds].to_numpy()

    def fundFeeNumpy(self):
        # fund1_index = list(self.df.columns).index('FundFee1')
        fund1_index = self.df.columns.get_loc("FundFee1")
        
        fee_np =self.df.iloc[:,fund1_index : fund1_index + self.num_of_funds].to_numpy()
        
        fee_np = np.zeros([10, 1])
        fee_np[0] = 0.003
        fee_np[1] = 0.005
        fee_np[2] = 0.006
        fee_np[3] = 0.008
        fee_np[4] = 0.001
        fee_np[5] = 0.0038
        fee_np[6] = 0.0045
        fee_np[7] = 0.0055
        fee_np[8] = 0.0057
        fee_np[9] = 0.0046
        # import pdb; pdb.set_trace()
        return fee_np.ravel()
    
    def getFundDFIndex(self, first_fund_name = 'FundValue1'):
        # import pdb; pdb.set_trace()
        # fund_index0 = list(self.df.columns).index('FundValue1')
        fund_index0 = self.df.columns.get_loc("FundValue1")
        fund_index1 = fund_index0 + self.num_of_funds
        return (fund_index0, fund_index1)
    
    def getFundData(self, first_fund_name = 'FundValue1'):
        idxs = self.getFundDFIndex(first_fund_name)
        df = self.df.iloc[:,idxs[0] : idxs[1]]
        return df.to_numpy()
        
    def calculateMappedIndex(self):
        mapping_mat = self.fund_mapping_matrix
        fund_mat = np.array(self.getFundData())
        # Map the fund into the index 
        return fund_mat.dot(mapping_mat.T)
    
    def randomSampleInforce(self, nsample):
        """Randomly sample the inforce

        Args:
            nsample (int): number of target inforces
        """
        return self.df.sample(n = nsample, random_state=1)

    def getInforceRecordId(self, record_id):
        """Randomly sample the inforce

        Args:
            nsample (int): number of target inforces
        """
        return self.df.iloc[record_id]
    
    def getFundDataRecordId(self, record_id):
        np_mat = self.getInforceRecordId(record_id).to_numpy()
        idxs = self.getFundDFIndex()
        return np_mat[idxs[0] : idxs[1]]
    
    def getMaxProjectionMonthsRecordId(self, record_id):
        df = self.getInforceRecordId(record_id)
        if 'ProjectionMonths' in df.keys():
            return df['ProjectionMonths']
        else:
            print('Invalid record id', record_id, ' max projection months is unknown')
        return -1
    
    def getInforceValidProjectionMat(self, proj_duration = 3, total_proj_points = 121):
        if_max_proj = np.ceil(self.df['ProjectionMonths'].to_numpy() / proj_duration).astype(np.int32)
        if_max_proj = np.minimum(if_max_proj, total_proj_points - 1)
        
        if_valid_proj_np = np.ones([len(if_max_proj), total_proj_points]) # (190k, 121)
        if_valid_proj_np_attain_age = np.zeros([len(if_max_proj), total_proj_points]) # (190k, 121)
        
        # import pdb; pdb.set_trace()
        for i in range(len(if_max_proj)):
            tp = if_max_proj[i]
            if_valid_proj_np_attain_age[i][tp] = 1
            if tp < 120:
                if_valid_proj_np[i][tp+1:] = 0
        return if_valid_proj_np, if_valid_proj_np_attain_age
            
        
        
    
# def unittest():
#     INFORCE_PATH = '/home/jzhang/github/nested_stochastic/data/inforce_input_full.csv'
#     SCENARIOS_PATH = '/home/jzhang/github/nested_stochastic/data/Scenarios_mini_qrtly.xlsx'
#     EXCEL_CALCULATION_PATH = '/home/jzhang/github/nested_stochastic/data/Excel Calculation_qrtly.xlsx'
#     fm = FundMapper(SCENARIOS_PATH)
#     inforce_ = Inforce(INFORCE_PATH, fm.getMappingMatrix())
#     import pdb; pdb.set_trace()


# unittest()