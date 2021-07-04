import numpy as np
import pandas as pd

class ScenarioLoader:
    """Load the scenarios from the excel 
    """
    def __init__(self, data_path, start_idx = 0, max_num_scenarios = 500, num_years = 30, num_of_funds = 10):
        self.df = pd.read_csv(data_path)
        mask = ~np.isnan(self.df['// Scenario'].to_numpy())
        self.num_datapoints = self.df[self.df['// Scenario'] == 1].shape[0]
        self.num_scenarios = max_num_scenarios
        self.start_idx = start_idx
        self.num_years = num_years
        self.num_of_funds = num_of_funds
    
    def numberOfScenarios(self):
        return self.num_scenarios

    def numberOfDataPoints(self):
        return self.num_datapoints
    
    def getScenario(self, scenarios_id = 0):
        # Input scenarios_id is is 0-indexed
        # SOA-ESG // Scenario is 1-indexed
        scenarios_id += 1 + self.start_idx
        df_tmp = self.df[self.df['// Scenario'] == scenarios_id]
        # The first col is the name of the scenarios
        data_ = df_tmp.iloc[:, 2: 2 + self.num_of_funds + 1].to_numpy()
        return data_

    

    