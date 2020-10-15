import numpy as np
import pandas as pd

class HMM:
    """
        Implementation of Filtering, Smoothing, Decoding(Viterbi) and Prediction
        for Hidden Markov Models
    """
    def __init__(self, T:np.ndarray, M:np.ndarray, state_list:list, obs_dict:dict):
        """
            Parameters:
            -----------
            T : transition probability matrix
                numpy array of shape [d,d] where d is number of states
                columns should sum to one
                        | S_{t-1}[0] | S_{t-1}[1] |
                S_{t}[0]|____________|____________|
                S_{t}[1]|____________|____________|

            M : observation probability matrix
                numpy array of shape [d,m] where m is the number of possible observations
                rows should sum to one
                        | O[1] | O[2] | ... | O[m] |
                S_{t}[0]|______|______| ... |______|
                S_{t}[1]|______|______| ... |______|

            state_list : list mapping states to row numbers

            obs_dict : dictionary mapping observations to column numbers

            Example:
            --------
            M = np.array([[1/6,1/6,1/6,1/6,1/6,1/6],
                          [1/10,1/10,1/10,1/10,1/10,1/2]])
            T = np.array([[0.95, 0.05],
                          [0.05, 0.95]])
            state_list = ['Fair','Loaded']
            meas_dict = {'1':0, '2': 1, '3':2, '4': 3, '5':4, '6': 5}

        """
        self.T = T
        self.M = M
        self.d = T.shape[0]
        self.m = M.shape[1]
        self.state_list = state_list
        self.obs_dict = obs_dict

    def filtering(self, obs:list, init_belief:np.ndarray):
        """
            Perform filtering on the given observation sequence
            Finds: P(S_t | o_{1:t})
            Parameters:
            -----------
            obs: list of observations
            init_belief: np.array of the initial belief of states

            Example:
            --------
            obs = ['1','2','4','6','6','6','3','6']
            init_belief = np.array([0.8,0.2])
        """
        # obs = ['1','2','4','6','6','6','3','6']
        assert len(init_belief) == self.d, "Initial belief should be for all possible states"
        assert np.sum(init_belief) == 1, "Sum of initial belief should be equal to 1"
        p_t = init_belief
        P_t = []
        P_t.append(init_belief)
        for i in range(len(obs)):
            p_t_non_norm = self.M[:,self.obs_dict[obs[i]]]*self.T.dot(p_t) 
            p_t = p_t_non_norm/np.sum(p_t_non_norm)
            P_t.append(p_t)
        return np.array(P_t)

    def smoothing(self, obs:list, init_belief:np.ndarray):
        """
            Perform smoothing on the sequence of observations
            Finds: P(S_{k}|o_{1:t}) for k<t
            Parameters:
            -----------
            obs: list of observations
            init_belief: np.array of the initial belief of states

            Example:
            --------
            obs = ['1','2','4','6','6','6','3','6']
            init_belief = np.array([0.8,0.2])
        """
        p_hat = self.filtering(obs, init_belief)
        b_kt = [] # will add in a reverse manner; have to flip upside down
        b_kt.append(np.array([1,1]))
        for i in range(len(obs)):
            b_mt = (b_kt[i] * self.M[:,self.obs_dict[obs[len(obs)-i-1]]]).dot(self.T) # (bmt*M[:,o]).T
            b_kt.append(b_mt)
        b_kt = np.flipud(b_kt) # flipping
        p_tilde_non_norm = b_kt*p_hat
        # normalise
        p_tilde = p_tilde_non_norm/(np.sum(p_tilde_non_norm,axis=1))[:,None]
        return p_hat, p_tilde

    def get_smoothing_table(self, obs:list, init_belief:np.ndarray):
        """
            Perform smoothing and filtering on the sequence of observations
            and print a table
            Parameters:
            -----------
            obs: list of observations
            init_belief: np.array of the initial belief of states

            Example:
            --------
            obs = ['1','2','4','6','6','6','3','6']
            init_belief = np.array([0.8,0.2])
        """
        p_hat, p_tilde = self.smoothing(obs,init_belief)
        p_hat_r = np.around(p_hat,4)
        p_tilde_r = np.around(p_tilde,4)
        df = pd.DataFrame()
        df['Observations'] = ['None']+obs
        for i in range(self.d):
            df[self.state_list[i]+'(filtered)'] = p_hat_r[:,i]
            df[self.state_list[i]+'(smoothing)'] = p_tilde_r[:,i]
        return df

    def decoding(self, obs:list, init_belief:np.ndarray):
        """
            Viterbi Algorithm
            -----------------
            Performs decoding on the sequence of states to get the 
            most likely trajectory
            Finds: s*_{0:t} = argmax_{s_{0:t}} P(S_{0:t}= s_{0:t}| o_{1:t})
            Parameters:
            -----------
            obs: list of observations
            init_belief: np.array of the initial belief of states

            Returns:
                The most likely path
                list of states

            Example:
            --------
            obs = ['1','2','4','6','6','6','3','6']
            init_belief = np.array([0.8,0.2])

            Returns: ['Loaded','Loaded','Loaded','Loaded','Loaded','Loaded','Loaded','Loaded',]
        """
        delta = init_belief
        deltas = []
        deltas.append(init_belief)
        max_delta_arg = [] # stores the argument of max(delta)
        for i in range(len(obs)):
            argmax_deltas = np.zeros(self.d)
            max_deltas = np.zeros(self.d)
            for j in range(self.d):
                path_probs = delta * self.T[j, :] * self.M[j, self.obs_dict[obs[i]]] # delta*P(st|st-1)*P(O|st)
                max_deltas[j] = np.max(path_probs)
                argmax_deltas[j] = np.argmax(path_probs)
            deltas.append(max_deltas)
            delta = max_deltas
            max_delta_arg.append(list(argmax_deltas))

        max_delta_args = np.array(max_delta_arg).astype(int)
        final_probs = deltas[-1]/np.sum(deltas[-1])
        prob_max_arg = np.argmax(final_probs)
        path = [self.state_list[int(prob_max_arg)]]
        len_path = len(max_delta_args)
        # backtrack
        for i in range(len_path-1):
            path.append(self.state_list[max_delta_args[len_path-1-i][int(prob_max_arg)]])
            prob_max_arg = max_delta_args[len_path-1-i][int(prob_max_arg)]

        return list(reversed(path))

    def get_decoding_table(self, obs:list, init_belief:np.ndarray):
        """
            Viterbi Algorithm
            -----------------
            Performs decoding on the sequence of states to get the 
            most likely trajectory
            Finds: s*_{0:t} = argmax_{s_{0:t}} P(S_{0:t}= s_{0:t}| o_{1:t})
            Parameters:
            -----------
            obs: list of observations
            init_belief: np.array of the initial belief of states

            Returns:
                The most likely path
                list of states

            Example:
            --------
            obs = ['1','2','4','6','6','6','3','6']
            init_belief = np.array([0.8,0.2])

            Returns: ['Loaded','Loaded','Loaded','Loaded','Loaded','Loaded','Loaded','Loaded',]
        """
        delta = init_belief
        deltas = []
        deltas.append(init_belief)
        max_delta_arg = [] # stores the argument of max(delta)
        for i in range(len(obs)):
            argmax_deltas = np.zeros(self.d)
            max_deltas = np.zeros(self.d)
            for j in range(self.d):
                path_probs = delta * self.T[j, :] * self.M[j, self.obs_dict[obs[i]]] # delta*P(st|st-1)*P(O|st)
                max_deltas[j] = np.max(path_probs)
                argmax_deltas[j] = np.argmax(path_probs)
            deltas.append(max_deltas)
            delta = max_deltas
            max_delta_arg.append(list(argmax_deltas))

        deltas = np.array(deltas)
        deltas = deltas/(np.sum(deltas,axis=1))[:,None]
        df = pd.DataFrame()
        df['Observation'] = ['None']+obs
        for i in range(self.d):
            df[self.state_list[i]] = deltas[:,i]
        return df

    def predict(self, k:int, obs:list, init_belief:np.ndarray):
        """
            Predict the probabilities of next-k states given a sequence of observations
            Finds: P(S_{t+k} | o_{1:t})
            Parameters:
            -----------
            k: the time step after t to predict
                basically find P(S_{t+k}|o_{1:t})
            obs: list of observations
            init_belief: np.array of the initial belief of states

            Returns:
                The most likely path
                list of states

            Example:
            --------
            k = 4
            obs = ['1','2','4','6','6','6','3','6']
            init_belief = np.array([0.8,0.2])
        """
        P_t = self.filtering(obs, init_belief)
        p_t = P_t[-1]
        p_tK =[]
        T = self.T.copy()
        for i in range(k):
            p_tK.append(T.dot(p_t))
            T = T.dot(T)
        return p_tK

    def get_prediction_table(self, k:int, obs:list, init_belief:np.ndarray):
        """
            Print the probabilities of next-k states given a sequence of observations
            Parameters:
            -----------
            k: the time step after t to predict
                basically find P(S_{t+k}|o_{1:t})
            obs: list of observations
            init_belief: np.array of the initial belief of states

            Returns:
                The most likely path
                list of states

            Example:
            --------
            k = 4
            obs = ['1','2','4','6','6','6','3','6']
            init_belief = np.array([0.8,0.2])
        """
        p_tK = np.array(self.predict(k, obs, init_belief))
        # print(p_tK)
        df = pd.DataFrame()
        time_steps = list(np.arange(1,k+1,1))
        df['Time Step'] = ['t+'+str(i) for i in time_steps]
        for i in range(self.d):
            df[self.state_list[i]+'(predictions)'] = p_tK[:,i]
        return df

if __name__ == "__main__":
    meas_dict = {'1':0, '2': 1, '3':2, '4': 3, '5':4, '6': 5} # mapping from measurement to columns
    state_list = ['Fair','Loaded']
    meas = ['1','2','4','6','6','6','3','6']
    init_belief = np.array([0.8,0.2])
    M = np.array([[1/6,1/6,1/6,1/6,1/6,1/6],
                [1/10,1/10,1/10,1/10,1/10,1/2]])
    T = np.array([[0.95, 0.05],
                [0.05, 0.95]])
    hmm = HMM(T, M, state_list, meas_dict)
    df = hmm.get_smoothing_table(meas, init_belief)
    path = hmm.decoding(meas, init_belief)
    df_pred = hmm.get_prediction_table(10,meas, init_belief)
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
    print('Predictions upto next 10 k steps')
    print(df_pred)
