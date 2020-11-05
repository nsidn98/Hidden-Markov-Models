from hmm import HMM
import numpy as np 

meas_dict = {'O1':0, 'O2': 1, 'O3':2}
state_list = ['S1','S2','S3']
meas = ['O2','O1','O3']
init_belief = np.array([1,0,0])
M = np.array([[0.1,0.2,0.7],
              [0.5,0.2,0.3],
              [0.3,0.5,0.2]])
T = np.array([[0.5, 0.2,0.1],
              [0.3, 0.2,0.7],
              [0.2, 0.4,0.2]])
# meas_dict = {'S':0, 'M': 1, 'L':2} # mapping from measurement to columns
# state_list = ['Hot', 'Cold']
# meas = ['S','M','S', 'L']
# init_belief = np.array([0.6,0.4])
# M = np.array([[0.1,0.4,0.5],
#               [0.7,0.2,0.1]])
# T = np.array([[0.7, 0.4],
#               [0.3, 0.6]])

hmm = HMM(T, M, state_list, meas_dict)
df = hmm.get_smoothing_table(meas, init_belief)
path = hmm.decoding(meas, init_belief)
df_pred = hmm.get_prediction_table(10,meas, init_belief)
df_decod = hmm.get_decoding_table(meas, init_belief)
df_decod_no = hmm.get_decoding_table(meas, init_belief,False)
print('#'*50)
print('Table of probabilities after smoothing and filtering')
print('#'*50)
print(df)
print()
print('#'*50)
print('Most likely trajectory followed')
print('#'*50)
print(path)
print()
print('#'*50)
print('Decoding beliefs')
print(df_decod)
print()
print('#'*50)
print('Decoding beliefs without norm')
print(df_decod_no)
print()
print('#'*50)
print('Deltas in decoding table')
df_decod_new = hmm.get_decoding_table_new(meas, init_belief,False)