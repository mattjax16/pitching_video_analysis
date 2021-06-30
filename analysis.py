'''analysis_old.py
Run statistical analyses and plot Numpy ndarray data
MATT BASS
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.style.use('seaborn')
#plt.style.use('ggplot')
#added these imports for extenstion
from data import *
import hashlib


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        if len(rows) != 0:
            return (np.amin(self.data.select_data(headers,rows), axis = 0))
        else:
            return (np.amin(self.data.select_data(headers), axis = 0))




    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        if len(rows) != 0:
            return (np.amax(self.data.select_data(headers, rows), axis=0))
        else:
            return (np.amax(self.data.select_data(headers), axis=0))

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        mins = self.min(headers,rows)

        maxes = self.max(headers, rows)

        return mins, maxes
    def mean(self, headers, rows=[], test_type=0):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        '''
        if test_type == 0:
           # I really dont need check for rows here
           selected_data = self.data.select_data(headers, rows)
           array_sum = selected_data.sum(axis=0)
           array_mean = array_sum/selected_data.shape[0]
           #print(selected_data.shape)
           return array_mean
        elif test_type == 1:

            if len(rows) != 0:
                return (np.mean(self.data.select_data(headers, rows), axis=0))
            else:
                return (np.mean(self.data.select_data(headers), axis=0))

    def var(self, headers, rows=[], test_type=0):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        '''

        if test_type == 0:

            #formulat is sum((x_i - mean of x)^2)/(total sample count - 1)

            selected_data = self.data.select_data(headers, rows)

            # sum((x_i - mean of x)^2) part
            array_mean = self.mean(headers,rows)


            #print(f'\n{selected_data.shape}\n{array_mean.shape}')

            top_part_start =(selected_data - array_mean)

            top_part_square = np.square(top_part_start)
            top_part = np.sum(top_part_square, axis=0)

            #complete formula and return
            return (top_part/(selected_data.shape[0]-1))
        if test_type == 1:

            selected_data = self.data.select_data(headers, rows)
            return np.var(selected_data,axis=0)


    def std(self, headers, rows=[], test_type=0):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        '''
        if test_type == 0:

            return (np.sqrt(self.var(headers,rows)))
        if test_type == 1:
            selected_data = self.data.select_data(headers, rows)
            return np.std(selected_data,axis=0)

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title=None, test_type=0, cat = None , fig_sz = (12,12)):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        if test_type == 0:
            headers = self.data.get_headers()
            if ind_var not in headers or dep_var not in headers:
                print(f"Error both ind_var : {ind_var} and dep_var : {dep_var}\nNeed to be in Headers:{self.data.get_headers()}")
                exit()
            else:
                if isinstance(self.data, AllData):
                    if cat == None:
                        var_array =  self.data.select_data([ind_var,dep_var])
                        x = var_array[:,0]
                        y = var_array[:,1]
                        plt.scatter(x,y)
                        plt.xlabel(ind_var)
                        plt.ylabel(dep_var)
                        plt.title(title)
                        return x,y
                    else:

                        if cat not in headers:
                            print(f"Error categorey(cat) : {cat}\nNeed to be in Headers:{self.data.get_headers()}")
                        else:
                            # data_cat_list = self.data.select_data([ind_var, dep_var, cat])
                            #
                            # var_array = data_cat_list[0]
                            cat_array = self.data.select_data([cat])
                            unique_cats_array = np.unique(cat_array)
                            color_array = np.array(list(map(self.colorHash, np.ndenumerate(unique_cats_array))))
                            # [-6289580265895947427  7212687553743170595   216846862356080747]
                            # print(color_array)
                            for label, color in zip(list(unique_cats_array), list(color_array)):
                                cat_rows_list = list(np.array(np.where(cat_array == label)).flatten())

                                plt.scatter(
                                    self.data.select_data(headers=[ind_var], rows=cat_rows_list),
                                    self.data.select_data(headers=[dep_var], rows=cat_rows_list),
                                    color=color, label=label)


                            plt.legend(loc='upper right')
                            plt.xlabel(f'{ind_var}')
                            plt.ylabel(f'{dep_var}')
                            plt.rcParams.update({'font.size': 20, 'figure.figsize': fig_sz})
                            if title == None:
                                plt.title(f'{ind_var} vs. {dep_var}')
                            else:
                                plt.title(title)


                else:
                    var_array = self.data.select_data([ind_var, dep_var])
                    x = var_array[:, 0]
                    y = var_array[:, 1]
                    plt.scatter(x, y)
                    plt.xlabel(ind_var)
                    plt.ylabel(dep_var)
                    if title == None:
                        plt.title(f'{ind_var} vs. {dep_var}')
                    else:
                        plt.title(title)
                    return x, y



    #helper function to get bottom kwd arg for matplotlib graphs
    def getBottom(self, labelList):

        if len(labelList) == 0:
            return None
        elif len(labelList) == 1:
            label_array = np.array(labelList).squeeze(axis=0)
            return label_array
        elif len(labelList) > 1:
            label_array = np.array(labelList)
            bottom_array = np.sum(label_array,axis=1).swapaxes(0,1)
            return bottom_array



    def histogram(self, var,title=None, test_type=0, cat = None):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        if test_type == 0:
            headers = self.data.get_headers()
            if var not in headers :
                print(f"Error var : {var}\nNeeds to be in Headers:{self.data.get_headers()}")
                exit()
            else:
                if isinstance(self.data, AllData):
                    if cat == None:
                        # var_array =  self.data.select_data([ind_var,dep_var])
                        # x = var_array[:,0]
                        # y = var_array[:,1]
                        # plt.scatter(x,y)
                        # plt.xlabel(ind_var)
                        # plt.ylabel(dep_var)
                        # plt.title(title)
                        # return x,y
                        pass
                    else:

                        if cat not in headers:
                            print(f"Error categorey(cat) : {cat}\nNeed to be in Headers:{self.data.get_headers()}")
                        else:
                            # data_cat_list = self.data.select_data([ind_var, dep_var, cat])
                            #
                            # var_array = data_cat_list[0]
                            cat_array = self.data.select_data([cat]).squeeze()
                            unique_cats_array = np.unique(cat_array)
                            color_array = np.array(list(map(self.colorHash, np.ndenumerate(unique_cats_array))))
                            # [-6289580265895947427  7212687553743170595   216846862356080747]
                            # print(color_array)
                            addedLabels = []
                            for label, color in zip(list(unique_cats_array), list(color_array)):
                                cat_rows_list = list(np.where(cat_array == label))[0]
                                cat_var_array = self.data.select_data(headers=[var], rows=cat_rows_list)

                                bottom_arg_val = self.getBottom(addedLabels)
                                plt.hist(cat_var_array,color=color, label=label, bottom=bottom_arg_val,  histtype='barstacked')

                                addedLabels.append(cat_var_array)

                else:
                   return

    def linePlot(self, var, title=None, test_type=0, cat=None):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        if test_type == 0:
            headers = self.data.get_headers()
            if var not in headers:
                print(f"Error var : {var}\nNeeds to be in Headers:{self.data.get_headers()}")
                exit()
            else:
                if isinstance(self.data, AllData):
                    if cat == None:
                        # var_array =  self.data.select_data([ind_var,dep_var])
                        # x = var_array[:,0]
                        # y = var_array[:,1]
                        # plt.scatter(x,y)
                        # plt.xlabel(ind_var)
                        # plt.ylabel(dep_var)
                        # plt.title(title)
                        # return x,y
                        pass
                    else:

                        if cat not in headers:
                            print(f"Error categorey(cat) : {cat}\nNeed to be in Headers:{self.data.get_headers()}")
                        else:
                            # data_cat_list = self.data.select_data([ind_var, dep_var, cat])
                            #
                            # var_array = data_cat_list[0]
                            cat_array = self.data.select_data([cat])
                            unique_cats_array = np.unique(cat_array)
                            color_array = np.array(list(map(self.colorHash, np.ndenumerate(unique_cats_array))))
                            # [-6289580265895947427  7212687553743170595   216846862356080747]
                            # print(color_array)
                            addedLabels = []
                            for label, color in zip(list(unique_cats_array), list(color_array)):
                                cat_rows_list = list(np.array(np.where(cat_array == label)).flatten())
                                cat_var_array = self.data.select_data(headers=[var], rows=cat_rows_list)

                                bottom_arg_val = self.getBottom(addedLabels)
                                plt.plot(cat_var_array, color=color, label=label,)

                                addedLabels.append(cat_var_array)

                            # plt.legend(loc='upper right')
                            # plt.xlabel(f'{ind_var}')
                            # plt.ylabel(f'{dep_var}')
                            # if title == None:
                            #     plt.title(f'{ind_var} vs. {dep_var}')
                            # else:
                            #     plt.title(title)

                            # x = var_array[:, 0]
                            # y = var_array[:, 1]
                            # plt.scatter(x, y)
                            # plt.xlabel(ind_var)
                            # plt.ylabel(dep_var)
                            # plt.title(title)
                            # return x, y

                else:
                    return


    #helper color hash function
    def colorHash(self,ndenumerate_obj):

        index = ndenumerate_obj[0][0]
        cat_label = str(ndenumerate_obj[1]).encode('utf-8')
        string_hash = hashlib.blake2s()
        string_hash.update(cat_label)

        hex_index_div = 1
        while index+12//(64*hex_index_div) >= 1:
                index = index//(64)
                hex_index_div+=1

        hex_index_div-= 1
        hex_index_div = 64 - hex_index_div
        color_hash_string_list = list(string_hash.hexdigest())
        # using List comprehension + isdigit() +split()
        # getting numbers from string
        numbers = []
        letters = []
        for i in color_hash_string_list:
            if i.isdigit():
                numbers.append(i)
            else:
                letters.append(i)
        letters = letters[len(letters):None:-1]
        offset = 1
        color_hex = f'{numbers[-1]}'
        # loop until color hex is len 6

        while len(color_hex) < 6 and int(color_hex) <= 16777215:
            color_hex+=str(numbers[len(color_hex) + offset])
            offset += 2

        color_hex = f'{int(color_hex):x}'

        if len(color_hex) >= 5:
            color_hex = color_hex[:4]
        color_hex_string = f'#{ color_hash_string_list[index] + color_hex + letters[-(index+1)]}'
        return color_hex_string
    #helper function for pair plot
    #take in the tuple from np.ndenumerate
    def createPairPlots(self, ndenumerate_obj, last_row, cat =None, diag = 'scatter', headers = [], fig = None):

        # get objects and headers

        ax = ndenumerate_obj[1]
        dependant_var = ndenumerate_obj[0][0]
        independant_var = ndenumerate_obj[0][1]
        headers = list(headers)

        if isinstance(self.data, AllData):

            # TODO: do for none
            if cat == None:
                ax.scatter(self.data.select_data([headers[independant_var]]), self.data.select_data([headers[dependant_var]]))
            else:
                #make colors based of of cat
                cat_array = (self.data.select_data([cat]).flatten())
                unique_cats_array = np.unique(cat_array)
                color_array = np.array(list(map(self.colorHash, np.ndenumerate(unique_cats_array))))
                # [-6289580265895947427  7212687553743170595   216846862356080747]
                #print(color_array)


                labels_list = []

                for label ,color in zip(list(unique_cats_array),list(color_array)):

                    cat_rows_list = list(np.array(np.where(cat_array == label)).flatten())



                    label1 = ax.scatter(self.data.select_data(headers = [headers[independant_var]], rows = cat_rows_list),
                           self.data.select_data(headers = [headers[dependant_var]],rows = cat_rows_list), label = f"{label} : {len(cat_rows_list)} samps")

                    labels_list.append(label1)



                # get rid of ticks
                if independant_var != 0 and dependant_var != last_row:
                    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                                       right=False,
                                       left=False,
                                       labelleft=False)
                elif dependant_var != last_row:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, right=False,
                                       left=False,
                                       labelleft=False)
                elif independant_var != 0:
                    ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, right=False,
                                       left=False,
                                       labelleft=False)



                #set diagnoals
                if independant_var == dependant_var:
                    ax.cla()
                    if diag == 'hist':
                        for label, color in zip(list(unique_cats_array), list(color_array)):
                            cat_rows_list = list(np.array(np.where(cat_array == label)).flatten())
                            #learned kwargs trick from https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/
                            kwargs = dict(alpha=0.5, bins=15)
                            ax.hist(self.data.select_data(headers=[headers[independant_var]], rows=cat_rows_list),
                                    **kwargs, label=f"{label} : {len(cat_rows_list)} samps", histtype ='barstacked')

                    elif diag == 'scatter':
                        for label, color in zip(list(unique_cats_array), list(color_array)):
                            cat_rows_list = list(np.array(np.where(cat_array == label)).flatten())
                            data = (self.data.select_data(headers=[headers[independant_var], cat], rows=cat_rows_list))
                            data_mean = self.mean([headers[independant_var]], cat_rows_list)[0]

                            kwargs = dict(alpha=0.5)
                            ax.scatter( data[:,1],np.sort(data[:,0]) ,label=str(label), **kwargs)

                            ax.scatter(data[0, 1], data_mean, color = 'black', marker = '*' , s = 130)


                    #added this for years and date times like that with lots of categories
                    elif diag == 'stats':
                        for label, color in zip(list(unique_cats_array), list(color_array)):
                            cat_rows_list = list(np.array(np.where(cat_array == label)).flatten())
                            data = (self.data.select_data(headers=[headers[independant_var], cat], rows=cat_rows_list))
                            data_mean = self.mean([headers[dependant_var]],cat_rows_list)


                            kwargs = dict(alpha=0.5)
                            ax.scatter(data[0, 1], data_mean, label=f"{label} : {len(cat_rows_list)} samps ")


                # if independant_var == 0 and dependant_var == 0:
                #
                #     ax.legend(handles = labels_list,loc = 'upper left',  bbox_to_anchor=(1.3, 0))
                # add labels only on outer edges
                if dependant_var == last_row:
                    ax.set_xlabel(headers[independant_var])

                if independant_var == 0:
                    ax.set_ylabel(headers[dependant_var])

                # have right ticks for scatters
                # if dependant_var == last_row and independant_var == last_row:
                #     yticks = ax.get_yticks()
                #     print(yticks)
                #     ax.set_xticks(yticks)

                fig.legend(handles=labels_list, loc='upper center', ncol=3)
                fig.subplots_adjust(top=0.92, bottom=0.08)
                return [ax,fig]
        else:

            ax.scatter(self.data.select_data([headers[independant_var]]), self.data.select_data([headers[dependant_var]]))



            #get rid of ticks
            if independant_var != 0 and dependant_var != last_row:
                ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                               left=False,
                               labelleft=False)
            elif dependant_var != last_row:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                         labelleft=False)
            elif independant_var != 0:
                ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                               labelleft=False)


            #add diagnoals:
            if independant_var == dependant_var and diag != 'scatter':
                ax.clear()
                if diag == 'hist':
                    ax.hist(self.data.select_data([headers[independant_var]]), edgecolor='black')

                elif diag == 'line':
                    ax.scatter(self.data.select_data([headers[independant_var]]), self.data.select_data([headers[dependant_var]]))


            if dependant_var == last_row:
                ax.set_xlabel(headers[independant_var])


            if independant_var == 0 :
                ax.set_ylabel(headers[dependant_var])


            return [ax, fig]

    def pair_plot(self, data_vars, fig_sz=(12, 12), title='', test_type=0, color_style='random', cat = None, diag = 'scatter'):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots

        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.

        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        if test_type == 0:
            # headers = self.data.get_headers()
            # data_vars = np.array([var for var in data_vars if var in headers])
            data_vars = np.array(data_vars)

            #had to take out share x and y for diagnoals
            fig, axs = plt.subplots(nrows=(data_vars.shape[0]), ncols=(data_vars.shape[0]), figsize = fig_sz, constrained_layout=False)
            #fig = fig.tight_layout(pad=5.0)

            axs_list = np.array(list((map(lambda ax: self.createPairPlots(ax, last_row = (axs.shape[0]-1), cat = cat, diag = diag, headers = data_vars, fig = fig), np.ndenumerate(axs)))))
            axs_array = axs_list[:,0]
            axs = axs_array.reshape(len(data_vars),len(data_vars))








            # for dep in range(axs.shape[0]):
            #     for independ in range(axs.shape[1]):
            #         axs[dep,independ].scatter(self.data.data[:,dep],self.data.data[:,independ])
            #         axs[dep, independ].set_xlabel(headers[dep])
            #         axs[dep, independ].set_ylabel(headers[independ])

            return fig, axs
