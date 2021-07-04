import pandas as pd
import  numpy as np

def get_fund_mapping_matrix(csv_path):
    df = pd.read_csv(csv_path)
    return df.iloc[0:10, 1:11].to_numpy()

def fund_remapping(scenario, mapping_matrix):
    """ Map from the invoice index to the unfd
    """
    return np.exp(scenario.dot(mapping_matrix))

def merge_rw_rn_scenario(rw_scenario, rn_scenario, switch_point):
    T = rw_scenario.shape[0]
    if switch_point < 1 or switch_point >= T:
        print('Invalid switch point {}'.format(switch_point))
        return None
    idx = switch_point
    left = rw_scenario[0:idx]
    num_rows = left.shape[1]
    # Get the second part of the RN data               
    right = rn_scenario[idx:][:,0:num_rows]
    return np.vstack([left, right])

def merge_rw_rn_scenario_full(rw_scenario, rn_scenario, switch_point0 = 1, switch_point1 = 120):
    """[summary]
    For each RW scenario (outter loop):
    The merged scenarios =
        no_merge: [rw_0, rw_1, rw_2, rw_3, rw_4,..., rw_30]
        merge1  : [rw_0, rn_1_1, rn_2_1, rn_3_1, rn_4_1,..., rn_30_1]
        merge2  : [rw_0, rw_1  , rn_2_2, rn_3_2, rn_4_2,..., rn_30_2]
        merge3  : [rw_0, rw_1  , rw_2  , rn_3_3, rn_4_3,..., rn_30_3]
        merge4  : [rw_0, rw_1  , rw_2  , rw3   , rn_4_4,..., rn_30_4]
    Args:
        startScenarioIdx (int, optional): [description]. Defaults to 2.
    """
    meged_scenarios = []
    for sp_t in list(range(switch_point0, switch_point1 + 1)):
        meged_scenarios.append(merge_rw_rn_scenario(rw_scenario, rn_scenario, sp_t))
    return np.array(meged_scenarios)
    
    
def random_select_inforces(target_inforce_size, min_inforce_id = 0, max_inforce_id = 1000000):
    # return np.array([6730, 1243, 5575, 3332, 9864, 5347, 8078, 9284, 8724, 6336, 3974,
    #    2443, 3607, 2151, 3931, 1445, 6975, 8500, 6927, 4490])
    return np.random.randint(min_inforce_id, max_inforce_id, target_inforce_size)
    