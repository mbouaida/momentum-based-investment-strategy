#dependencies
import numpy as np
import pandas as pd
from scipy import optimize
np.random.seed(2008)

class Optimization():
    """
    Optimize portfolio with constraints
    """
    
    def __init__(self, df: pd.DataFrame, weekMomentum: dict):
        self.df = df
        self.weekMomentum = weekMomentum
        
    def biCriterionFunctionOptmzn(self, MeanReturns, CovarReturns, RiskAversParam, weekSelection): 
        """
        Function to handle bi-criterion portfolio optimization with constraints
        ----------
        **Parameters:
        MeanReturns: np.array
        CovarReturns: np.array
        RiskAversParam: np.array
        RiskAversParam: float
        weekSelection: pd.DataFrame
        **Returns:
        -------
        scipy.OptimizeResult
        """

        PortfolioSize = weekSelection.shape[0]

        def A_matrix(weekSelection):
            A = []
            for c in weekSelection.cluster:
                inds= [i for i, x in enumerate(weekSelection.cluster) if x == c]
                l = list(np.zeros(PortfolioSize))
                for ind in inds:
                    l[ind] = 1
                A.append(l)    
            return(A)


        def  f(x, MeanReturns, CovarReturns, RiskAversParam, PortfolioSize=PortfolioSize):
            PortfolioVariance = np.matmul(np.matmul(x, CovarReturns), x.T) 
            PortfolioExpReturn = np.matmul(np.array(MeanReturns),x.T)
            func = RiskAversParam * PortfolioVariance - (1-RiskAversParam)*PortfolioExpReturn
            return func

        def ConstraintEq(x):
            A=np.ones(x.shape)
            b=1
            constraintVal = np.matmul(A,x.T)-b 
            return constraintVal

        def ConstraintIneqUpBounds(x):
            A= A_matrix(weekSelection)
            bUpBounds =np.array([0.3]*PortfolioSize).T
            constraintValUpBounds = bUpBounds-np.matmul(A,x.T) 
            return constraintValUpBounds

        def ConstraintIneqLowBounds(x):
            A= A_matrix(weekSelection)
            bLowBounds =np.array([0.05]*PortfolioSize).T
            constraintValLowBounds = np.matmul(A,x.T)-bLowBounds  
            return constraintValLowBounds

        xinit=np.repeat(0.01, PortfolioSize)
        cons = ({'type': 'eq', 'fun':ConstraintEq}, \
            {'type':'ineq', 'fun': ConstraintIneqUpBounds},\
            {'type':'ineq', 'fun': ConstraintIneqLowBounds})
        bnds = [(0,1)]*PortfolioSize

        opt = optimize.minimize(f, x0 = xinit, args = ( MeanReturns, CovarReturns,\
                                                    RiskAversParam, PortfolioSize), \
                             method = 'SLSQP',  bounds = bnds, constraints = cons, \
                             tol = 10**-3)
        print(opt)
        return opt
    
    def defiReturnsComputing(self, DefiPrice, Rows, Columns):
        """
        Function to compute Defi returns
        ----------
        **Parameters:
        DefiPrice : np.asarray
        Rows: int
        Columns: int
        **Returns:
        -------
        np.array
        """ 
        
        DefiReturn = np.zeros([Rows-1, Columns])
        for j in range(Columns):  # j: Assets
            for i in range(Rows-1):     #i: Daily Prices
                DefiReturn[i,j]=((DefiPrice[i+1, j]-DefiPrice[i,j])/DefiPrice[i,j])

        return DefiReturn    

    def getOptimalAllocation(self):
        """
        Function to get optimal allucation for a given week
        ----------
        **Parameters:
        **Returns:
        -------
        dict
        """     
        
        #extract asset labels
        assetLabels = self.df.columns
        print('Asset labels for portfolio : \n', assetLabels)

        #compute asset returns
        arDefiPrices = np.asarray(self.df)
        [Rows, Cols]=arDefiPrices.shape
        arReturns = self.defiReturnsComputing(arDefiPrices, Rows, Cols)

        weeklyRet = []
        for defi in range(10):
            weekRet = 1
            for r in [l[defi] for l in arReturns[-7:]]:
                weekRet *= 1+r
            weeklyRet.append(weekRet-1)


        #compute mean returns and variance covariance matrix of returns
        meanReturns = np.mean(arReturns, axis = 0)
        covReturns = np.cov(arReturns, rowvar=False)

        #initialization
        xOptimal =[]
        minRiskPoint = []
        expPortfolioReturnPoint =[]
        week_date = self.df.index[-1]

        for points in range(0,30):
            riskAversParam = points/30.0
            result = self.biCriterionFunctionOptmzn(meanReturns, covReturns, riskAversParam, \
                                               self.weekMomentum[week_date])
            xOptimal.append(result.x)

        #compute annualized risk and return  of the optimal portfolios for trading days = 251  
        xOptimalArray = np.array(xOptimal)
        minRiskPoint = np.diagonal(np.matmul((np.matmul(xOptimalArray,covReturns)),\
                                             np.transpose(xOptimalArray)))
        riskPoint =   np.sqrt(minRiskPoint*251) 
        expPortfolioReturnPoint= np.matmul(xOptimalArray, meanReturns )
        retPoint = 251*np.array(expPortfolioReturnPoint)

        # get best combine
        ret_risk = retPoint/riskPoint
        f_int = lambda v : ((v+0.25)//0.5)*0.5
        ret_risk = list(f_int(ret_risk))[::-1]
        ind_best = ret_risk.index(max(ret_risk))
        return(list(assetLabels),list(xOptimalArray[-(ind_best+1)]),list(weeklyRet))