# make-pickle-mergertree

## Extract mainbranches
[extract_mainbranch.py](extract_mainbranch.py) extracts mainbranches from MergerTree.  
You have to change some variables in [extract_mainbranch.py](extract_mainbranch.py).  
Below are the steps.
1. Specify a file name of MergerTree(.tree) with ***NAME_DF*** variable.
2. Specify name and column-index of parameters to extract with ***PARAM_NAME_LIST*** and ***PARAM_IDX_LIST***.<br>
Note: Match name and column-index of parameters to MergerTree.
3. Specify a mass-range of extract haro with ***RANGE_MIN*** and ***RANGE_MAX***.
4. Execute [extract_mainbranch.py](extract_mainbranch.py).

Thus, mainbranches of hosthalo and subhalo are saved in .csv format.<br>

The [host_param.pickle](/make-pickle-mergertree/sample-params/host_param.pickle)'s data-struct is *{"file_name":{"parameter_name":ndrray(1-dim)}}*.  
Key *"filename"* is defined by the filename of the file from which extracted mainbranches(In the [sample-params](/make-pickle-mergertree/sample-params), mainbranch_0_0_0.csv).  
Key *parameter_name* is MergerTree's parameter.  
The value of *"parameter_name"* is time-series-data of parameter.
The [sub_param.pickle](/make-pickle-mergertree/sample-params/sub_param.pickle)'s data-struct is *{"file_name":{"parameter_name":**[ndrray(1-dim)]**}}*.  
Key *filename* and *parameter_name* are the same as [host_param.pickle](/make-pickle-mergertree/sample-params/host_param.pickle), but the value of *parameter_name* is list-struct with *ndarray(1-dim)*. Each *ndarray(1-dim)* is a different subhalo's time-series-data.
<br>