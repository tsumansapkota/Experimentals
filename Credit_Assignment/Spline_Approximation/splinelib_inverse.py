import numpy as np


class Spline1D(object):

    def __init__(self, max_points, x, y, epsilon=0.1, ): #x,y for initialization
        self.n_points = max_points # changes dynamically
        self.n_max = max_points # max point is constant
        self.eps = epsilon
        
        X_ = np.random.uniform(x.min()+epsilon, x.max()-epsilon, size=(max_points-2))
        X = np.empty(shape=(self.n_points))
        X[1:-1] = X_
        X[0], X[-1] = x.min()-epsilon, x.max()+epsilon
        self.X = X

        self.Y = np.random.uniform(y.min(), y.max(), size=(max_points))
        self.rangeX = None
        self.rangeY = None
        self.rangeX_n = None
        self.diffX = None
        self.diffY = None

        self.input = x
        self.output = None

        self.del_output = None
        self.del_X = None
        self.del_Y = None
        self.del_input=None

        self.preprocess()
        pass

 ####################################################################################   
 ####################################################################################

    def _inrange_(self, X, break0, break1): #if x is after

        xmsk1 = X >= break0
        xmsk2 = X < break1
        xmsk = np.bitwise_and(xmsk1, xmsk2)
        xs = xmsk #*X
        return xs

    def _sort_parameters_(self,):
        sortindx = np.argsort(self.X)
        self.X = self.X[sortindx]
        self.Y = self.Y[sortindx]

    def _calculate_rangeX_(self,):

        rangeX = np.zeros((self.n_points-1, self.input.shape[0]))

        for i in range(self.n_points-1):
            rangeX[i] = self._inrange_(self.input, self.X[i], self.X[i+1])
        self.rangeX = rangeX

        rnx_ = np.count_nonzero(rangeX, axis=1)
        rangeX_n = np.zeros(self.n_points)
        rangeX_n[:-1] += rnx_
        rangeX_n[1:] += rnx_
        rangeX_n[rangeX_n == 0.] = -1.

        self.rangeX_n = rangeX_n
        return self.rangeX


    def _calculate_rangeY_(self, y):

        rangeY = np.zeros((self.n_points-1, self.input.shape[0]))

        for i in range(self.n_points-1):
            rangeY[i] = self._inrange_(y, self.Y[i], self.Y[i+1])
        self.rangeY = rangeY
        # rnx_ = np.count_nonzero(rangeY, axis=1)
        # rangeX_n = np.zeros(self.n_points)
        # rangeX_n[:-1] += rnx_
        # rangeX_n[1:] += rnx_
        # rangeX_n[rangeX_n == 0.] = -1.
        # self.rangeX_n = rangeX_n
        return self.rangeY

    def preprocess(self,):

        self._sort_parameters_()
        self._calculate_rangeX_()

        self.diffX = np.diff(self.X)
        self.diffY = np.diff(self.Y)


    def forward(self,input):
        self.input = input
        self.preprocess()

        output = np.zeros_like(self.input)
        for i in range(self.n_points-1):
            Y_ = self.diffY[i]/self.diffX[i] *(self.input - self.X[i]) + self.Y[i]
            output = output + Y_*self.rangeX[i]
        self.output = output
        return self.output

 ##################################################################################   

    def _backward_Y_(self,):

        consts = np.zeros((self.n_points-1, self.input.shape[0]))
        for i in range(self.n_points-1):
            consts[i] = (self.input-self.X[i])/self.diffX[i]

        dY = np.zeros((self.n_points, self.input.shape[0]))
        
        dY[0] = (-1* consts[0] +1) *self.rangeX[0]
        dY[-1] = consts[-1] *self.rangeX[-1]

        for i in range(1, self.n_points-1):
                a = consts[i-1]*self.rangeX[i-1]
                b = (-1* consts[i] +1)*self.rangeX[i]
                dY[i] = a+b
        dY = dY*self.del_output
        dY_= dY.sum(axis=1)/self.rangeX_n
        dY = dY.mean(axis=1)
        # dY = dY.sum(axis=1)/self.rangeX_n
        dY[0], dY[-1] = dY_[0], dY_[-1]

        self.del_Y = dY
        return self.del_Y

    def _backward_X_(self,):

        consts = np.zeros((self.n_points-1, self.input.shape[0]))
        for i in range(self.n_points-1):
            consts[i] = self.diffY[i]/(self.diffX[i]**2)
        
        dX = np.zeros((self.n_points, self.input.shape[0]))
        dX[0] = consts[0]*(self.input - self.X[1])*self.rangeX[0]
        dX[-1] = -1*consts[-1]*(self.input - self.X[-2])*self.rangeX[-1]

        for i in range(1, self.n_points-1):
                a = -1*consts[i-1]*(self.input - self.X[i-1])*self.rangeX[i-1]
                b = consts[i]*(self.input - self.X[i+1])*self.rangeX[i]
                dX[i] = a+b
        dX = dX*self.del_output
        ##########This is true delX#############
        #Not Implemented
        ########################################
        dX = dX.mean(axis=1)
        # dX = dX.sum(axis=1)/self.rangeX_n

        self.del_X = dX
        return self.del_X

    def _backward_input_(self,):

            dinp = np.zeros_like(self.input)
            for i in range(self.n_points-1):
                dinp = dinp + self.diffY[i]/self.diffX[i] *self.rangeX[i]
            
            dinp = dinp*self.del_output
            self.del_input = dinp
            return self.del_input

    def backward(self, del_output):
        self.del_output = del_output
        self._backward_Y_()
        self._backward_X_()
        self._backward_input_()
        return self.del_input

####################################################################################

    def update(self, learning_rate=0.1):
        self.X = self.X - self.del_X*learning_rate
        self.Y = self.Y - self.del_Y*learning_rate
        self._sort_parameters_()

        self.X[0] = self.input.min()-self.eps
        self.X[-1] = self.input.max()+self.eps



 ##################################################################################   

    def _remove_close_points_(self, min_dist=1e-3):
        # removing ones which are very close to each other
        # requires sorted points first
        x_diff = np.ones_like(self.X)
        x_diff[1:] = np.diff(self.X)
        clipmask = np.abs(x_diff) > min_dist
        self.X = self.X[clipmask]
        self.Y = self.Y[clipmask]
        self.n_points = len(self.X)
    
    def _combine_linear_points_(self, min_area=1e-2):
        triangle = np.ones_like(self.X)
        for i in range(self.n_points-2):
            triangle[i+1] = 0.5*np.abs(
                (self.X[i] - self.X[i+2])*self.diffY[i]+self.diffX[i]*(self.Y[i+2] - self.Y[i]))
        mergemask = triangle > min_area
        self.X = self.X[mergemask]
        self.Y = self.Y[mergemask]
        self.n_points = len(self.X)

    def  _add_new_point_(self, min_error=1e-4):
        # adding units where the error > min_error
        if self.n_points < self.n_max:
            dYs = np.zeros((self.n_points-1, self.input.shape[0]))
            for i in range(self.n_points-1):
                dYs[i] = self.del_output * self.rangeX[i]
            dYerr = (dYs**2).mean(axis=1)
            index = np.argmax(dYerr)
            if dYerr[index] > min_error:
                newpx = (self.X[index] + self.X[index+1])/2.
                newpy = (self.Y[index] + self.Y[index+1])/2.
                # adding new interpolation points
                self.X = np.append(self.X, newpx)
                self.Y = np.append(self.Y, newpy)
                # sorting the points for plotting
                self.n_points = len(self.X)
                self._sort_parameters_()
    
    def _remove_no_input_points_(self,):
        #removing if points contain no input
        self.preprocess()
        nx = np.zeros_like(self.X)
        nx_ = np.count_nonzero(self.rangeX, axis=1)
        nx[:-1] += nx_
        nx[1:] += nx_

        nx0mask = nx!=0
        self.X = self.X[nx0mask]
        self.Y = self.Y[nx0mask]
        self.n_points = len(self.X)