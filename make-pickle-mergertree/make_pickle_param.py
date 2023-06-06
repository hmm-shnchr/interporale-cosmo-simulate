import numpy as np
import pandas as pd
import pickle
import sys

######################### CONSTANTS #########################
READ_COLS       = 60
COL_NAMES       = ["ch{0:02d}".format(i) for i in range(READ_COLS)]

MAINBRANCH_LIST = ["mainbranch_0_0_0.csv"] ## filename to read (change as appropriate)
RANGE_MIN       = "1e+7"
RANGE_MAX       = "1e+18"
## Specify the column number of  parameters to be extract as a list.
## See MergerTree about the correspondence between column number and parameter.
PARAM_IDX_LIST  = [0, 1, 5, 10, 11, 17, 20]
PARAM_NAME_LIST = ["ScaleFactor", "ID", "pid", "Mvir", "Rvir", "x", "vx"]

PICKLE_NAME = "_" + RANGE_MIN + "_" + RANGE_MAX
for p_name in PARAM_NAME_LIST:
    PICKLE_NAME += "_" + p_name
PICKLE_NAME += ".pickle"

class GetTreeInfo:
    """
    Get some information of mainbranch.
    """
    def __init__(self, df):
        self.df         = df
        self.tree_index = np.where(df=="tree")[0]

    def getParam(self, ch):
        """
        Get the parameter specified by ch.

        Parameters
        ----------
        ch: int
            number of the column to get

        Returns
        ----------
        return_list: list
            each index corresponds to a halo-parameter
        """
        return_list = []
        for t_idx in range(len(self.tree_index)):
            if t_idx != len(self.tree_index)-1:
                current_tree = self.df[self.tree_index[t_idx]+1:self.tree_index[t_idx+1]]
            else:
                current_tree = self.df[self.tree_index[t_idx]+1:]

            info_list = []
            for current_halo in range(current_tree.shape[0]):
                info_list.append(float(current_tree[current_halo, ch]))
            return_list.append(info_list)

        return return_list

def classified_index(classify_list, min_val, max_val):
    index_list = []
    for idx, elem in enumerate(classify_list):
        if elem >= min_val and elem <= max_val:
            index_list.append(idx)
    return index_list

if __name__ == "__main__":
    df_dict = {}
    for key in MAINBRANCH_LIST:
        try:
            df_dict[key] = np.array(pd.read_csv(key, names = COL_NAMES, dtype = str))
        except:
            print("Cannot read {key}.".format(key))
            sys.exit(1)

    param_dict      = {}
    for i in range(len(PARAM_IDX_LIST)):
        param_dict[PARAM_NAME_LIST[i]] = PARAM_IDX_LIST[i]

    ## Extract the specified parameters by param_dict.
    getTreeInfoDict = {}
    param           = {}
    for key in MAINBRANCH_LIST:
        getTreeInfoDict[key] = GetTreeInfo(df_dict[key])
    for param_key in PARAM_NAME_LIST:
        param[param_key] = {}
        for m_key in MAINBRANCH_LIST:
            param[param_key][m_key] = getTreeInfoDict[m_key].getParam(param_dict[param_key])

    ## Extract Mvir(z=0) of all haloes to get use_idx_dict.
    use_idx_dict = None
    if "Mvir" in PARAM_NAME_LIST:
        mvir_z0 = {}
        for m_key in MAINBRANCH_LIST:
            mvir_z0_list = []
            for idx in range(len(param["Mvir"][m_key])):
                mvir_z0_list.append(param["Mvir"][m_key][idx][-1])
            mvir_z0[m_key] = mvir_z0_list

        use_idx_dict = {}
        for m_key in MAINBRANCH_LIST:
            use_idx_dict[m_key] = classified_index(mvir_z0[m_key], float(RANGE_MIN), float(RANGE_MAX))
            print("m_key : {},  length : {}".format(m_key, len(use_idx_dict[m_key])))

    m_str = ""
    for m_key in MAINBRANCH_LIST:
        m_str += m_key[11:16] + "_"

    ## Extract a host halo.
    host_param = {}
    for m_key in MAINBRANCH_LIST:
        host_param[m_key] = {}
        for p_key in PARAM_NAME_LIST:
            host_param[m_key][p_key] = np.array(param[param_key][m_key][0])

    with open("host_param.pickle", mode = "wb") as f:
        pickle.dump(host_param, f)

    ## Extract sub haloes.
    sub_param = {}
    for m_key in MAINBRANCH_LIST:
        sub_param[m_key] = {}
        for p_key in PARAM_NAME_LIST:
            sub_param[m_key][p_key] = []
            if use_idx_dict is not None:
                for use_idx in use_idx_dict[m_key]:
                    if use_idx == 0: continue
                    sub_param[m_key][p_key].append(np.array(param[p_key][m_key][use_idx]))
            else:
                for idx, sub_p in enumerate(param[p_key][m_key]):
                    if idx == 0: continue
                    sub_param[m_key][p_key].append(np.array(sub_p))

    with open("sub_param.pickle", mode = "wb") as f:
        pickle.dump(sub_param, f)