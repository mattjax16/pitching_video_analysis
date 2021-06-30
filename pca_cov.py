'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
YOUR NAME HERE
CS 251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA centered
        self.Ac = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        #A_mean: ndarray(num_selected_vars,)
        # Holds the mean of A
        self.A_mean = None

        #undo_normilazation: ndarray (steps, num_vars)
        # Contains the values (row-wise going first to last) of the data needed to undo the normalization
        #TODO maybe add operdator data key value pair
        self.undo_normilazation = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs


    def get_A_mean(self):
        '''(No changes should be needed)'''
        return self.A_mean

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''

        self.A_mean = data.mean(axis=0)
        Ac = data - self.A_mean

        #set self.Ac
        self.Ac = Ac

        #compute cov matrix
        cov_matrix = (Ac.T@Ac)/(data.shape[0]-1)
        return cov_matrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        return list(e_vals/e_vals.sum())

    def compute_cum_var(self, prop_var, method = 'loop'):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_var_list = []
        if method == 'loop':
            for i in range(0,len(prop_var)):
                cum_var_list.append(np.array(prop_var[:(i+1)]).sum())
            return cum_var_list






    #helper method to normalize data sepreatly when preforming PCA:
    def normalize_separately(self, A):
        A_min = A.min(axis = 0)
        A_range = A.max(axis = 0) - A_min

        self.undo_normilazation = np.stack((A_range,A_min), axis = 0)
        norm_A_first_part = (A-A_min)
        if A_range.max() != 0 and A_range.min() != 0:
            A = norm_A_first_part/A_range
        return A


    def pca(self, vars, normalize=False, norm_method = 'separately'):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.
        norm_method: string
            By default is seperatly, decides which normalization method to use if normalize is True

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''

        #check that all vars are in self.data
        for var in vars:
            if var not in self.data.columns.values:
                print(f'Error!!! var:\n\t{var}\nNeed to be in:\n\t{self.data.columns.values}')
                return
        self.vars = vars
        self.A = self.data[vars].values

        # check to see if data needs to be normalized
        if normalize:
            self.normalized = True
            # Check to see which normalization method to do and do it
            if norm_method == 'separately':
                self.A = self.normalize_separately(self.A)


        #Get the covariance matrix (A is centerd in the method)
        cov_matrix = self.covariance_matrix(self.A)

        # Get eigenvalues and eigenvectors
        self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)

        # Get Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = self.compute_prop_var(self.e_vals)

        # cum_var: Python list. len(cum_var) = num_pcs
        # Get Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = self.compute_cum_var(self.prop_var)

        return

    def elbow_plot(self, num_pcs_to_keep=None,markersize='7',x_label_size = 10, figsize = (10,10),show_just_numbers = False, show_final_k_percentage = False):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if isinstance(num_pcs_to_keep, type(None)):
            fig, ax = plt.subplots(1, 1,figsize =figsize)
            num_PCs = [(i + 1) for i in range(len(self.cum_var))]
            ax.plot(num_PCs,self.cum_var, marker = 'o', markersize=markersize, markerfacecolor='red')
            ax.set_xticks(num_PCs)
            if not show_just_numbers:
                ax.set_xticklabels([f'K{i}' for i in num_PCs], fontsize=x_label_size)
            ax.set_xlabel("Principle Components (Dimensions)")
            ax.set_ylabel("Cumulative Variance Explained")
            ax.set_title(f"Cumulative Variance Explained\nFor All Principle Components")
        else:
            fig, ax = plt.subplots(1, 1,figsize =figsize)
            num_PCs = [(i + 1) for i in range(len(self.cum_var[:num_pcs_to_keep]))]
            ax.plot(num_PCs, self.cum_var[:num_pcs_to_keep], marker='o', markersize=markersize, markerfacecolor='red')
            ax.set_xticks(num_PCs)
            if not show_just_numbers:
                ax.set_xticklabels([f'K{i}' for i in num_PCs], fontsize = x_label_size)
            ax.set_xlabel("Principle Components (Dimensions)")
            ax.set_ylabel("Cumulative Variance Explained")
            ax.set_title(f"Cumulative Variance Explained\nFor The Top {num_pcs_to_keep}\nPrinciple Components")
            if show_final_k_percentage:
                ax.annotate(f'{self.cum_var[num_pcs_to_keep-1]: .4f}', (num_PCs[-1], self.cum_var[:num_pcs_to_keep][-1]))

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''

        #check for correct pcs_to_keep
        if len(pcs_to_keep) > len(self.e_vals):
            print(f'Error There Are More PCs To Keep Than PCs Avalible')
            exit()
        if max(pcs_to_keep) > len(self.e_vals):
            print(f'Error The Max PC to Keep ({max(pcs_to_keep)}) is to high\n'+
                  f'Max it can be is {len(self.e_vals)-1}')
            exit()
        if min(pcs_to_keep) < 0:
            print(f'Error Min Pc to keep is {min(pcs_to_keep)}\n'+
                  f'Smallest it can be is 0')
            exit()

        # make sure pcs_to_keep is an array
        pcs_to_keep = np.ravel(np.array(pcs_to_keep))

        #make sure all pcs_to_keep are ints
        #Ran into a lot of errors since it says in parameters will be a list and in project
        # an array is passed in so after lots of trial an error I came across this genius method
        # from here https://stackoverflow.com/questions/934616/how-do-i-find-out-if-a-numpy-array-contains-integers
        # which I fit to meet my needs for this project
        if not all(np.equal(np.mod(pcs_to_keep, 1), 0)):
            print(f'Error!! All PCs Need to be Ints, They are currently:'
                  + f'\n\t{pcs_to_keep}')

        #select the pcs_to_keep for P hat
        P_hat = self.e_vecs[:,pcs_to_keep]
        A_proj = self.Ac @ P_hat
        self.A_proj = A_proj
        return A_proj



    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        A_reconstructed: ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''

        # Check that PCA has been done
        if isinstance(self.prop_var, type(None)):
            print(f'Error PCA needs to be done!!!!!!')
            exit()

        # check that top_k isn't greater than the amount of PCs
        if top_k > len(self.prop_var):
            print(f'Error there are not {top_k} PCs!!!'
                  +f'\nIn total there are {len(self.prop_var)} PCs!')
            exit()


        A_means = np.mean(self.A)
        pcs_to_keep = np.arange(top_k)
        A_reconstructed = (self.pca_project(pcs_to_keep)) @ (self.e_vecs[:,pcs_to_keep]).T

        if self.normalized:
            A_reconstructed = self.undo_normilazation[0,:].T * A_reconstructed

        A_reconstructed = A_reconstructed + A_means
        return A_reconstructed


