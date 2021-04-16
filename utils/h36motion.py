from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import utils.forward_kinematics as fk
import torch
import utils.data_utils as data_utils
import os
import pickle as pkl


class H36motion(Dataset):
    def __init__(self, path_to_data, actions, input_n=10, output_n=10, dct_n=20, split=0, sample_rate=2, use_dct=True, train_3d=False):
        """
        read h36m data to get the dct coefficients.
        :param path_to_data:
        :param actions: actions to read
        :param input_n: past frame length
        :param output_n: future frame length
        :param dct_n: number of dct coeff. used
        :param split: 0 train, 1 test, 2 validation
        :param sample_rate: 2
        :param data_mean: mean of expmap
        :param data_std: standard deviation of expmap
        """
        print('NOTE THAT WE HAVE REMOVED DATA MEAN AND DATA STD')
        self.dct_n = dct_n
        self.input_n = input_n
        self.output_n = output_n
        self.path_to_data = path_to_data
        self.use_dct = use_dct
        self.split = split
        self.train_3d = train_3d
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])

        acts = data_utils.define_actions(actions)

        subjs = subs[split]
        self.sequences_expmap, self.sequences_3d, self.all_seqs = self.load_data(subjs, acts, sample_rate, input_n + output_n, input_n=input_n)
        self.all_seqs = np.concatenate(self.all_seqs, 0)

        self.reduced_seqs_expmap = self.sequences_expmap[:,:,self.dimensions_to_use]
        self.reduced_seqs_3d = self.sequences_3d[:,:,self.dimensions_to_use_3d]

        if use_dct and self.train_3d:
            self.input_dct_seq, self.output_dct_seq = self.get_dct(self.reduced_seqs_3d)
        elif use_dct and not self.train_3d:
            self.input_dct_seq, self.output_dct_seq = self.get_dct(self.reduced_seqs_expmap)

    def get_dct(self, seqs, seq_are_3d=False):
        if seq_are_3d:
            dims_to_use = self.dimensions_to_use_3d
        else:
            dims_to_use = self.dimensions_to_use
        seqs = seqs.transpose(0, 2, 1)
        seqs = seqs.reshape(-1, self.input_n + self.output_n)
        seqs = seqs.transpose()
        dct_m_in, _ = data_utils.get_dct_matrix(self.input_n + self.output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(self.input_n + self.output_n)

        # padding the observed sequence so that it has the same length as observed + future sequence
        pad_idx = np.repeat([self.input_n - 1], self.output_n)
        i_idx = np.append(np.arange(0, self.input_n), pad_idx)
        input_dct_seq = np.matmul(dct_m_in[:self.dct_n, :], seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dims_to_use), self.dct_n])

        output_dct_seq = np.matmul(dct_m_out[:self.dct_n], seqs)
        output_dct_seq = output_dct_seq.transpose().reshape([-1, len(dims_to_use), self.dct_n])

        return input_dct_seq, output_dct_seq

    def read_sequence(self, subject, action, subaction, sample_rate):
        print("Reading subject {0}, action {1}, subaction {2}".format(subject, action, subaction))

        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subject, action, subaction)
        sequence = data_utils.readCSVasFloat(filename)

        sampled_sequence = sequence[::sample_rate, :]
        num_frames = len(sampled_sequence)

        return sequence, num_frames

    def get_subsequence(self, sequence, num_frames, seq_len):
        fs = np.arange(0, num_frames - seq_len + 1)
        fs_sel = fs
        for i in np.arange(seq_len - 1):
            fs_sel = np.vstack((fs_sel, fs + i + 1))
        fs_sel = fs_sel.transpose()
        seq_sel = sequence[fs_sel, :]

        return seq_sel

    def find_indices_srnn(self, frame_num1, frame_num2, seq_len, input_n=10):
        """
        Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

        which originaly from
        In order to find the same action indices as in SRNN.
        https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState(SEED)

        T1 = frame_num1 - 150
        T2 = frame_num2 - 150  # seq_len
        idxo1 = None
        idxo2 = None
        for _ in np.arange(0, 4):
            idx_ran1 = rng.randint(16, T1)
            idx_ran2 = rng.randint(16, T2)
            idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
            idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
            if idxo1 is None:
                idxo1 = idxs1
                idxo2 = idxs2
            else:
                idxo1 = np.vstack((idxo1, idxs1))
                idxo2 = np.vstack((idxo2, idxs2))
        return idxo1, idxo2

    def load_data(self, subjects, actions, sample_rate, seq_len, input_n=10):
        """
        adapted from
        https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216

        :param seq_len: past frame length + future frame length
        :param is_norm: normalize the expmap or not
        :param input_n: past frame length
        :return:
        """
        cache_name = os.path.join(self.path_to_data, '_'.join(['learn_traj3', str(subjects), str(actions), str(sample_rate), str(seq_len), str(input_n)]) + '.pkl')

        if os.path.isfile(cache_name):
            print('loading data from cache: {}'.format(cache_name))
            sequences_expmap, sequences_3d, complete_seq, sampled_seq = pkl.load(open(cache_name, 'rb'))
        else:
            sampled_seq, complete_seq = [], []
            for subj in subjects:
                for action in actions:
                    sequence1, num_frames1 = self.read_sequence(subj, action, 1, sample_rate)
                    sequence2, num_frames2 = self.read_sequence(subj, action, 2, sample_rate)

                    if subj == 5:
                        # subject 5 is the testing subject, we use a specific scheme to extract the frame idxs
                        # such that they are the same as in related work
                        fs_sel1, fs_sel2 = self.find_indices_srnn(num_frames1, num_frames2, seq_len, input_n=input_n)
                        seq_sel1 = sequence1[fs_sel1, :]
                        seq_sel2 = sequence2[fs_sel2, :]
                    else:
                        seq_sel1 = self.get_subsequence(sequence1, num_frames1, seq_len)
                        seq_sel2 = self.get_subsequence(sequence2, num_frames2, seq_len)

                    sampled_seq.append(seq_sel1), sampled_seq.append(seq_sel2)
                    complete_seq.append(sequence1), complete_seq.append(sequence2)

            sequences_expmap = np.concatenate(sampled_seq, axis=0)
            complete_seq = np.concatenate(complete_seq, axis=0)

        zeroed = sequences_expmap.copy()
        zeroed[:, :, 0:6] = 0
        sequences_3d = H36motion.expmap2xyz(zeroed)

        self.data_std = np.std(complete_seq, axis=0)
        self.data_mean = np.mean(complete_seq, axis=0)

        self.dimensions_to_ignore, self.dimensions_to_use = [], []
        self.dimensions_to_ignore.extend(list(np.where(self.data_std < 1e-4)[0]))
        self.dimensions_to_use.extend(list(np.where(self.data_std >= 1e-4)[0]))
        self.data_std[self.dimensions_to_ignore] = 1.0
        self.data_mean[self.dimensions_to_ignore] = 0.0
        # first 6 elements are global translation and global rotation
        self.dimensions_to_use = self.dimensions_to_use[6:]

        joint_to_ignore_3d = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        self.dimensions_to_ignore_3d = np.concatenate((joint_to_ignore_3d * 3, joint_to_ignore_3d * 3 + 1, joint_to_ignore_3d * 3 + 2))
        self.dimensions_to_use_3d = np.setdiff1d(np.arange(sequences_3d.shape[-1]), self.dimensions_to_ignore_3d)

        print('Saving data to cache: {}...'.format(cache_name))
        pkl.dump([sequences_expmap, sequences_3d, complete_seq, sampled_seq], open(cache_name, 'wb'))

        return sequences_expmap, sequences_3d, sampled_seq

    @staticmethod
    def expmap2xyz(expmap):
        """
        convert expmaps to joint locations
        """
        shape_in = expmap.shape
        if len(shape_in) == 3:
            expmap = expmap.reshape(shape_in[0]*shape_in[1], -1)

        parent, offset, rotInd, expmapInd = fk._some_variables()
        if isinstance(expmap, torch.Tensor):
            xyz = fk.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
        else:
            xyz = fk.fkl_torch(torch.from_numpy(expmap), parent, offset, rotInd, expmapInd)

        if len(shape_in) == 3:
            xyz = xyz.reshape(shape_in[0], shape_in[1], -1)

        return xyz

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]

def test_visualization():
    from torch.utils.data import DataLoader
    acts = data_utils.define_actions('walking')
    data_dir = '/run/media/bob/ext/human_exponential_format/h3.6m/dataset/'

    test_dataset = H36motion(path_to_data=data_dir, actions=acts[0], input_n=10, output_n=25, split=1, sample_rate=2, dct_n=35, use_dct=False, train_3d=True)
    loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    for batch in loader:
        _, all_seqs = batch

        fig = plt.figure()
        ax = plt.gca(projection='3d')

        plt.cla()
        viz.plot_predictions(all_seqs[0, :, :], all_seqs[0, :, :], fig, ax, 'Pose', is_3d=True)
        plt.pause(1)


if __name__ == '__main__':
    test_visualization()