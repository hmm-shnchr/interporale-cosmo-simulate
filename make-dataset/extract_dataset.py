import numpy as np


def extract_dataset(mainbranch, parameter, host_param, sub_param, ext_data, acc_sf, input_size, output_size, threshold, box_size):
    dataset = {}
    for m_key in mainbranch:
        dataset[m_key] = {}
        for p_key in parameter:
            dataset[m_key][p_key] = []
            for idx, p in enumerate(sub_param[m_key][p_key]):
                if "Mvir" in parameter:
                    if sub_param[m_key]["Mvir"][idx][-1] < threshold: continue

                start_i = 0
                if ext_data in ["After_acc", "All_acc"]:
                    if acc_sf[m_key][idx] == -1: continue
                    if ext_data == "After_acc":
                        start_i         = acc_sf[m_key][idx]
                        unity_size      = input_size + output_size
                        surplus_size    = p[start_i:].size % unity_size
                        add_size        = unity_size - surplus_size + input_size
                        if (start_i - add_size) >=0:
                            start_i -= add_size

                ## Extract parameter.
                if p_key in ["Mvir"]:
                    data = np.log(p[start_i:] / p[acc_sf[m_key][idx]])
                else:
                    data = p[start_i:]

                dataset[m_key][p_key].append(data)

    return dataset