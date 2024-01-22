import numpy as np

my_data_pth = "data/hall6.npz"
my_data = np.load(my_data_pth, allow_pickle=True)
my_data = dict(my_data)

total_ex_num = 0
for key in my_data.keys():
    print(key)
    total_ex_num+=len(my_data[key].item(0).keys())
    print(my_data[key].item(0).keys())
    print('n_views : '+str(len(my_data[key].item(0)['task0_example0'])))
    print('n_frames : '+str(len(my_data[key].item(0)['task0_example0'][0])))
    print(my_data[key].item(0)['task0_example0'][0][0][0])
print("Total number of examples: ", total_ex_num)