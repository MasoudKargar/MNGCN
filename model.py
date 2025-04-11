from MNGCN import MNGCN


def get_model(param):
    
    if param['model'] == 'MNGCN':
        model = MNGCN(param)
    

    return model