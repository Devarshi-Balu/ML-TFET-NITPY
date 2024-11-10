# files in the directory 
    # 1.Train_code_bagged_extra_tree_go_collab.html 
        a.This file contains the trained code file in html format, it contains the code and graphs plotted
    # 2.train_code_bagged_ExtraaTreeRegressor.ipynb 
        a.This file contains the final and actual code used to train the model. 
# Parameter considerations
    1.  Tsi is ignored since it is same across all the experiments
    2.  At some parts of the code "Ignore Nc" might be observed - but actually Tsi is ignored, Nc is taken into account and 
        it is not ignored anywhere.
        Nc was ignored at inital stages to check the performance of the model without Nc, but later the approach is continued,
        Nc is taken into account.

# points to be noted 
    1. Donot train the model using this code on a Laptop the code will crash
    2. Donot load the trained model in laptop - minimum ram requirement is above 20GB
    3. While training make sure that n_jobs = -1 , to use all the cores in the machine 
    
