from tqdm import tqdm
import pandas as pd
import csv

NAME_DF     = "tree_0_0_0.dat" ## filename to read (change as appropriate)
######################### CONSTANTS #########################
READ_COLS   = 60
COL_NAMES   = ["ch{0:02d}".format(i) for i in range(READ_COLS)]
OPEN_FNAME  = "mainbranch_0_0_0.csv"

if __name__ == "__main__":
    df = pd.read_csv(NAME_DF, names = COL_NAMES, delim_whitespace = True, dtype = str)

    ## Get indices of ch00 == "#tree".
    tree_index = list(df.reset_index().query("ch00 == '#tree'").index)
    print("halo-num : {hnum}".format(hnum=len(tree_index)))

    file_mainbranch     = open(OPEN_FNAME, mode = "a", newline = "")
    writer_mainbranch   = csv.writer(file_mainbranch)

    for t_idx in tqdm(range(len(tree_index))):
        if t_idx != len(tree_index)-1:
            current_df = df[tree_index[t_idx]:tree_index[t_idx+1]].reset_index()
        else:
            current_df = df[tree_index[t_idx]:].reset_index()

        ## Get depth_first_id(ch28) of a last halo in a mainbranch from the root halo.
        last_mainleaf_depthfirst_id = current_df.at[current_df.index[1], "ch34"]

        ## Sort current_df by Depth_first_ID in descending-order.
        current_df=current_df.sort_values(by=["ch28"], ascending=False)
        current_df=current_df.drop(columns=["index"])

        ## Write a mainbranch.
        writer_mainbranch.writerow(["tree"])
        for idx, item in current_df.iterrows():
            if item["ch28"] <= last_mainleaf_depthfirst_id:
                writer_mainbranch.writerow(item.to_list())
                if item["ch03"] == "-1": break

    file_mainbranch.close()
