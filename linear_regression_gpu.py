'''linear_regression_old.py
Subclass of Analysis that performs linear regression on data
YOUR NAME HERE
CS251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys

import analysis


#for gpu aceleration
import cupy as cp

class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data,use_gpu = False):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression

        #############################3333
        # Question does a have 1's column
        ##################################3
        self.A = None

        # predicted_y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.predicted_y = None

        # actual_y: ndarray. shape=(num_data_samps, 1)
        #   Vector for the real dependent variables in the data set
        self.actual_y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.m_sse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

        # ind_vars: list of the string of the independant vars
        self.ind_vars = None
        # dep_var: string. dependant variable
        self.dep_var = None

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = cp
        else:
            self.xp = np

    # helper functions for gpu acceleration
    def checkArrayType(self, data):
        if self.use_gpu:
            if cp.get_array_module(data) == np:
                data = cp.array(data)
        else:
            if cp.get_array_module(data) == cp:
                data = np.array(data)

        return data

    # helper function to get things as numpy
    def getAsNumpy(self, data):
        if cp.get_array_module(data) == cp:
            data = data.get()
        return data

    # helper function to get things as numpy
    def getAsCupy(self, data):
        if cp.get_array_module(data) == np:
            data = cp.array(data)
        return data

    def linear_regression(self, ind_vars = None, dep_var = None, method='scipy', p=1, slope = None, intercept = None):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function. (DEFAULT)
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).
         p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).
             !!!! FOR SINGLE IND_VAR ONLY!!!!!

        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
            Initilized at None and only sets the slope to skip the regression if set
            (ie without regression for fitting)
        intercept: float.
            Intercept for the linear regression fit
            Initilized at None and only sets the slope to skip the regression if set
            (ie without regression for fitting)
        !! both slope and intercept need to be set for it to fit and not run a regression

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        # first I am checking ind_vars and dep_vars and making sure they are entered and exist as headers

        if isinstance(ind_vars, type(None)) or isinstance(dep_var, type(None)):
            print(f'Error: there must be atleast 1 ind_var and dep_var\nRight now they are {ind_vars} and  {dep_var}')
            sys.exit()
        if len(ind_vars) < 1:
            print(f'Error: there must be at least 1 ind_var')
            sys.exit()

        ind_vars_array = self.xp.array(ind_vars)
        headers_array = self.xp.array(self.data.get_headers())

        if dep_var not in headers_array:
            print(f'Error: dep_var: {dep_var} needs to be in {headers_array}')
            sys.exit()
        for ind_var in ind_vars_array:
            if ind_var not in headers_array:
                print(f'Error: ind_var: {ind_var} needs to be in {headers_array}')
                sys.exit()

        #make sure ther is only one independant variable if p is greater than 1
        if p > 1 and len(ind_vars) > 1:
            print(f'ERROR: Can only have one ind_var if p is greater than 1\n'
                  f'Currently there are {len(ind_vars)} ind_vars')




        # Set the variables list
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        # NOW i AM "Use your data object to select the variable columns associated with the
        # independent and dependent variable strings."

        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(self.ind_vars)
        self.actual_y = self.data.select_data([self.dep_var])
        self.p = p

        if not isinstance(slope, type(None)) and  not isinstance(intercept, type(None)):
            #check for errors when fitting with input sizes
            if p == 1 and len(slope) != len(ind_vars):
                print(f'ERROR: there must be {len(ind_vars)} slopes\n'
                      f'Currently there are {len(slope)} slopes')
                exit()
            if p > 1 and len(slope) != p:
                print(f'ERROR: there must be {len(ind_vars)} slopes\n'
                      f'Currently there are {len(slope)} slopes')
                exit()

            self.slope = slope
            self.intercept = intercept
        else:
            # Now I am copying how they do it in cs252 because I worked ahead on this project from
            # the one from last year
            # so here I am going to do the linear regeression based on the method chosen
            if method == 'scipy':
                if p == 1:
                    c = self.linear_regression_scipy(self.A, self.actual_y)
                elif p >1:
                    poly_mat = self.make_polynomial_matrix(self.A,self.p)
                    c = self.linear_regression_scipy(poly_mat, self.actual_y)
                self.slope = c[1:]
                self.intercept = float(c[0])
            if method == 'normal':
                if p == 1:
                    c = self.linear_regression_normal(self.A, self.actual_y)
                elif p >1:
                    poly_mat = self.make_polynomial_matrix(self.A,self.p)
                    c = self.linear_regression_normal(poly_mat, self.actual_y)
                self.slope = c[1:]
                self.intercept = float(c[0])
            if method == 'qr':
                if p == 1:
                    c = self.linear_regression_qr(self.A, self.actual_y)
                elif p >1:
                    # poly_mat = self.make_polynomial_matrix(self.A,self.p)
                    # c = self.linear_regression_normal(poly_mat, self.y)
                    print(f"Error: P must = 1 it is {p}")
                self.slope = c[1:]
                self.intercept = float(c[0])


        self.predicted_y = self.predict()
        self.residuals = self.compute_residuals(self.predicted_y)
        self.R2 = self.r_squared(self.predicted_y)
        self.m_sse = self.mean_sse()




    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''

        #chech inputs are proper array types
        A = self.checkArrayType(A)
        y = self.checkArrayType(y)


        A = self.xp.hstack([self.xp.ones([A.shape[0], 1]), A])
        c, res, rnk, s = scipy.linalg.lstsq(A, y)
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''

        # chech inputs are proper array types
        A = self.checkArrayType(A)
        y = self.checkArrayType(y)

        A = self.xp.hstack([self.xp.ones([A.shape[0], 1]), A])

        A_first_part = (A.T@A)
        invA = self.xp.linalg.inv(A_first_part)

        second_part = (A.T)@(y)

        c = invA@second_part

        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''

        # chech inputs are proper array types
        A = self.checkArrayType(A)
        y = self.checkArrayType(y)

        A = self.xp.hstack([self.xp.ones([A.shape[0], 1]), A])
        Q, R = self.qr_decomposition(A)
        c = scipy.linalg.solve_triangular(R,(Q.T@y))
        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''

        # chech inputs are proper array types
        A = self.checkArrayType(A)

        num_rows, num_cols = A.shape
        Q = self.xp.zeros((num_rows, num_cols))
        for j in range(num_cols):
            u = A[:, j]
            for i in range(j):
                # (Q[:, i]@u) * Q[:, i] is the projection
                u = u - (Q[:, i] @ u) * Q[:, i]
            Q[:, j] = u / self.xp.linalg.norm(u)

        R = Q.T @ A
        return Q, R


    def predict(self, X=None, method = 'vectorized'):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''


        if isinstance(X, type(None)):
            X = self.A


        #make sure X is right size shape=(num_data_samps, num_ind_vars)
        #question does order matter
        if X.shape[0] != self.A.shape[1]:
            if X.shape[0] != self.A.shape[0]:
                print(f"Error: X Has Shape {X.shape} it Needs {self.A.shape[1]} Independant Variables")
                sys.exit()
        if self.p > 1:
            X = self.make_polynomial_matrix(X,self.p)


        # make sure x is the right array type
        X = self.checkArrayType(X)

        if method == "vectorized":

            y_pred = self.xp.array(self.intercept + self.xp.sum((X*self.xp.array(self.slope).T),1))[:,self.xp.newaxis]
            return y_pred

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''

        # make sure y_pred is the right array type
        y_pred = self.checkArrayType(y_pred)

        y_mean = self.xp.mean(self.actual_y)
        r2_bottom = self.xp.sum((self.actual_y - y_mean) ** 2)
        r2_top = self.xp.sum((self.predicted_y - y_mean)**2)
        R2 = (r2_top/r2_bottom)
        return R2

    def compute_residuals(self, y_pred):

        #why not make self. here
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''

        # make sure y_pred is the right array type
        y_pred = self.checkArrayType(y_pred)
        self.actual_y = self.checkArrayType(self.actual_y)

        residuals = self.actual_y - y_pred
        return residuals



    def mean_sse(self):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        m_sse = (self.xp.sum((self.residuals)**2))/(len(self.residuals))
        return m_sse

    def scatter(self, ind_var, dep_var, title = None):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        if isinstance(title, type(None)):
            title = f'R^2 : {self.R2:.8F}'
        else:
            title = f'{title}\nR^2 : {self.R2:.8F}'

        x_cords, y_cords = analysis.Analysis.scatter(self, ind_var=ind_var, dep_var=dep_var, title=title)

        #make sure cors are numpy
        x_cords = self.getAsNumpy(x_cords)
        y_cords = self.getAsNumpy(y_cords)

        x_linreg_plot_cords = np.linspace(np.min(x_cords),np.max(x_cords))
        if self.p == 1:
            y_linreg_plot_cords = self.intercept + self.slope[self.ind_vars.index(ind_var)]*x_linreg_plot_cords
            plt.plot(x_linreg_plot_cords, y_linreg_plot_cords, label=f'linear reg line (p={self.p})', c='r', alpha=0.5)
        elif self.p > 1:
            x_linreg_A = x_linreg_plot_cords[:,np.newaxis]
            x_linreg_A = self.make_polynomial_matrix(x_linreg_A,self.p)
            y_linreg_plot_cords =np.array(self.intercept + np.sum((x_linreg_A*np.array(self.slope).T),1))[:,np.newaxis]
            plt.plot(x_linreg_plot_cords, y_linreg_plot_cords, label=f'poly reg line (p={self.p})', c='r',
                    alpha=0.5)

        plt.legend(bbox_to_anchor=(1.26, 0.1), loc='upper right')



    # helper function for pair_plot to create linear regressions
    def pair_plot_linear_regs(self, ndenumerate_obj, diag = 'scatter'):

        ax = ndenumerate_obj[1]
        dependant_var = ndenumerate_obj[0][0]
        independant_var = ndenumerate_obj[0][1]
        headers = self.data.get_headers()
        x = headers[independant_var]
        y = headers[dependant_var]
        self.linear_regression(ind_vars=[y], dep_var=x)

        x_cords = self.data.select_data([x])

        x_linreg_plot_cords = np.linspace(np.min(x_cords), np.max(x_cords))
        y_linreg_plot_cords = self.intercept + self.slope[0] * x_linreg_plot_cords

        if diag == 'scatter':
            if self.p == 1:
                ax.plot(x_linreg_plot_cords, y_linreg_plot_cords, label=f'linear reg line (p={self.p})', c='r', alpha=0.5)
            if self.p > 1:
                ax.plot(x_linreg_plot_cords, y_linreg_plot_cords, label=f'poly reg line (p={self.p})', c='r', alpha=0.5)
            ax.set_title(f'R^2 : {self.R2:.8F}')
        elif diag == 'hist':
            if independant_var != dependant_var:
                if self.p == 1:
                    ax.plot(x_linreg_plot_cords, y_linreg_plot_cords, label=f'linear reg line (p={self.p})', c='r',
                            alpha=0.5)
                if self.p > 1:
                    ax.plot(x_linreg_plot_cords, y_linreg_plot_cords, label=f'poly reg line (p={self.p})', c='r',
                            alpha=0.5)
                ax.set_title(f'R^2 : {self.R2:.8F}')

        return ax




    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''

        if hists_on_diag == True:
            fig, axes = analysis.Analysis.pair_plot(self, data_vars=data_vars, fig_sz=fig_sz, title='', diag='hist')
            # here I am mappinng the regression lines onto all the scatter plots in the pair plot
            # with the function I made pair_plot_linear_regs()
            axes_list = np.array(list((map(lambda ax: self.pair_plot_linear_regs(ax, diag='hist'),
                                           np.ndenumerate(axes)))))
        else:
            fig, axes = analysis.Analysis.pair_plot(self, data_vars=data_vars, fig_sz=fig_sz, title='', diag='scatter')
            # here I am mappinng the regression lines onto all the scatter plots in the pair plot
            # with the function I made pair_plot_linear_regs()
            axes_list = np.array(list((map(lambda ax: self.pair_plot_linear_regs(ax),
                                           np.ndenumerate(axes)))))

        axes_array = axes_list
        axes = axes_array.reshape(len(data_vars), len(data_vars))

        return

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        #check that A is right shape
        if A.shape[1] != 1:
            print(f"ERROR: A needs one independent variable only\nCurrently it has {A.shape[1]} ")
        # if A.shape[1] != 1:
        #     print(f"ERROR: A needs {self.residuals.shape[0]} Samples\nCurrently it has {A.shape[1]} ")

        #get right array type for A
        A = self.checkArrayType(A)

        poly_mat = self.xp.power(A*(self.xp.ones((A.shape[0],p))),(1+self.xp.arange(p)))
        return (poly_mat)

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        # I just added this to linear regression
        pass

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return  self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p = 1, use_gpu = False):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_var: list or str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
            If multiple ind_vars it is a list
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor.
        '''
        # This function is going to call linear regression
        # here I am seeing if just one ind_var is used and if it is I am putting it into a list
        if isinstance(ind_vars, type(str)):
            ind_vars = [ind_vars]

        #fit the data
        self.linear_regression(ind_vars=ind_vars,dep_var=dep_var,
                               p = p, slope=slope,intercept=intercept)
