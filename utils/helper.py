import os


def create_exp_folder(exp, model_name):
    exp_root_folder = 'runs'
    exp_folder_name = f'{exp}_{model_name}'
    path = exp_root_folder + os.sep + exp_folder_name
    
    if not os.path.isdir(exp_root_folder): # create if root exp folder not exist
        os.mkdir(exp_root_folder)
    if not os.path.isdir(path): # create if exp folder does not exist
        os.mkdir(path)
        
    return path