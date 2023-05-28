import numpy as np

def load_params():
    #load param_vals.npy from current directory
    file_path = 'param_vals.npy'
    with open(file_path, 'rb') as f:
        param_vals = np.load(f,allow_pickle=True)

    param_vals = list(param_vals)
    del param_vals[848]
    return param_vals

def to_txt_3d(out):
    permuted = out.permute(0,2,3,4,1).detach().cpu().numpy()
    with open("debug_outputs/torch_debug.npy", "wb") as f:
        np.save(f, permuted)
        
    # #all possible length 3 combinations of 10 and 20 
    indices = [[10,10,10],[10,10,20],[10,20,10],[10,20,20],[20,10,10],[20,10,20],[20,20,10],[20,20,20]]
    with open("torch_debug.txt", "w") as f:
        f.write('Mean: %s \n' % np.mean(permuted))
        f.write('Standard deviation: %s \n' % np.std(permuted))
        for i,j,k in indices:
            f.write('index: %s,%s,%s' % (i,j,k))
            f.write('\n')
            f.write(str(permuted[0,i,j,k,:8]))
            f.write('\n')

def to_txt_1d(out):
    with open("debug_outputs/torch_debug.npy", "wb") as f:
        np.save(f, out.detach().cpu().numpy())

def to_txt(out):
    print('TORCH SHAPE:', out.shape)
    if len(out.shape)==2:
        to_txt_1d(out)
        return
    if len(out.shape)==5:
        to_txt_3d(out)
        return
    permuted = out.permute(0,2,3,1).detach().cpu().numpy()
    with open("debug_outputs/torch_debug.npy", "wb") as f:
        np.save(f, permuted)
        
    # #all possible length 3 combinations of 20 and 40
    # indices = [[20,20,20],[20,20,40],[20,40,20],[20,40,40],[40,20,20],[40,20,40],[40,40,20],[40,40,40]]
    #all possible length 2 combinations of 10 and 20 
    indices = [[10,10],[10,20],[20,10],[20,20]]
    with open("torch_debug.txt", "w") as f:
        f.write('Mean: %s \n' % np.mean(permuted))
        f.write('Standard deviation: %s \n' % np.std(permuted))
        for i,j in indices:
            f.write('index: %s,%s' % (i,j))
            f.write('\n')
            f.write(str(permuted[0,i,j,:8]))
            f.write('\n')
