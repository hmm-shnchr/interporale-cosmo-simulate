import pandas as pd
import csv

NAME_DF     = "MW032.tree" ## filename to read (change as appropriate)
######################### CONSTANTS #########################
READ_COLS   = 61
COL_NAMES   = ["ch{0:02d}".format(i) for i in range(READ_COLS)]
OPEN_FNAME  = "mainbranch_" + NAME_DF[:-5] + ".csv"

if __name__ == "__main__":
    df = pd.read_csv(NAME_DF, names = COL_NAMES, delim_whitespace = True, dtype = str)

    ## get row number of ch00 == "#tree"
    tree_index = list(df.reset_index().query("ch00 == '#tree'").index)
    print(len(tree_index))

    file_mainbranch     = open(OPEN_FNAME, mode = "a", newline = "")
    writer_mainbranch   = csv.writer(file_mainbranch)

    for t_idx in range(len(tree_index)):
        if t_idx != len(tree_index)-1:
            current_df = df[tree_index[t_idx]:tree_index[t_idx+1]].reset_index()
        else:
            current_df = df[tree_index[t_idx]:].reset_index()

        ## get depth_first_id(ch28) of a last halo in a main branch from the root halo
        last_mainleaf_depthfirst_id = str(current_df.at[current_df.index[1], "ch34"])
        
        ## go to the last halo in the main branch with depth_first_id(ch28)
        current_halo = int(list(current_df.query("ch28 == @last_mainleaf_depthfirst_id").index)[0])

        writer_mainbranch.writerow(["tree"])
        ## get the main branch from the last halo with desc_id(ch03) and id(ch01)
        while True:
            current_data = []
            ## save each parameters
            for col in COL_NAMES:
                current_data.append(current_df.at[current_df.index[current_halo], col])
            writer_mainbranch.writerow(current_data)
            
            ## ch03 == -1 indicates root halo in the tree
            if current_df.at[current_df.index[current_halo], "ch03"] == "-1":   break

            ## desc_id indicates an id(ch01) of descendant halo
            desc_id         = str(current_df.at[current_df.index[current_halo], "ch03"])
            current_halo    = int(list(current_df.query("ch01 == @desc_id").index)[0])
            if current_halo == 0:   current_halo += 1

    file_mainbranch.close()
