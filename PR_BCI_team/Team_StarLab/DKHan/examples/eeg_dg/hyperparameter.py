
# model = 'Deep4net'
model = 'Resnet'
# model = 'Deep4net_mtl'
# model = 'MSNN'
# model = 'EEGNet'
# model = 'SampleEEGNet'

# data_path = '/x_data_440.npy'
# data_path = '/x_data_380.npy'
# data_path = '/x_data_380_norm.npy'
# data_path = '/x_data_440_norm.npy'
data_path = '/x_data_830_e.npy'




# data_path = '/x_data_440_e.npy'

batch_size = 256
lr =0.005 * batch_size/1024
# lr =0.005 * batch_size/32
# lr=0.001/4

# lr=3