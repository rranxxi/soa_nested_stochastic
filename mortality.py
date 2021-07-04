import numpy as np
import pandas as pd

class Mortality:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.mortality_rate = self.df.to_numpy()[1:][:,1:3].astype(np.float64)
        self.attained_ages = self.df.to_numpy()[1:][:,0].astype(np.float64)
    
    def mortalityRateFemale(self):
        return self.mortality_rate[:,1]
        
    def mortalityRateMale(self):
        return self.mortality_rate[:,0]
    
    def attainedAge(self):
        return self.attained_ages
    
    def getMortalityRateFromAge(self, attain_age, num_years, valuation_freq = 4, is_male = True):
        """Get the mortality rate and surviveship from a provided attain age

        Args:
            attain_age ([type]): [description]
            num_years ([type]): [description]
            is_male (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        rate = None
        if is_male:
            rate = self.mortalityRateMale()
        else:
            rate = self.mortalityRateFemale()
        rate = rate[attain_age:]
        if len(rate) < num_years + 1:
            rate = np.pad(rate, num_years + 1 - len(rate), mode='edge')
        rate2 = rate[0: num_years + 1] / (1.0 * valuation_freq)
        rate3 = np.repeat(rate2, valuation_freq)
        rate3[0] = 0
        rate3 = rate3[0:1 + num_years * valuation_freq]
        
        # Compute the survivorship from the mortality rate
        survivorship = np.zeros_like(rate3)
        survivorship[0] = 1.0
        
        for i in range(1, len(rate3)):
            survivorship[i] = survivorship[i - 1] * (1 - rate3[i])
        return rate3, survivorship
        
    def getInforceMortalitySurvivor(self, if_age, if_sex, num_years, valuation_freq = 4):
        num_of_inforce = len(if_age)
        num_or_proj = num_years * valuation_freq + 1
        if_mortality = np.zeros([num_of_inforce, num_or_proj])
        if_survivorship = np.zeros([num_of_inforce, num_or_proj])
        for i in range(num_of_inforce):
            is_male = if_sex[i] == 'M'
            # import pdb; pdb.set_trace()
            r, s = self.getMortalityRateFromAge(if_age[i], num_years, valuation_freq = valuation_freq, is_male = is_male)
            if_mortality[i] = r
            if_survivorship[i] = s
        # shape (190k, 121)
        return if_mortality, if_survivorship
        
def unittest():
    EXCEL_CALCULATION_PATH = '/home/jzhang/github/nested_stochastic/data/Excel Calculation_qrtly2.xlsx'
    INFORCE_PATH = '/home/jzhang/github/nested_stochastic/data/inforce_input_full.csv'
    df = pd.read_csv(INFORCE_PATH)
    df.columns = df.columns.str.replace(' ', '')
    if_sex = df['gender'].to_numpy()
    if_age = df['AttainedAge'].to_numpy().astype(np.int32)
    
    mr = Mortality(EXCEL_CALCULATION_PATH) 
    ifr, ifs = mr.getInforceMortalitySurvivor(if_age, if_sex, 30, 4)
    import pdb; pdb.set_trace()

# unittest()
