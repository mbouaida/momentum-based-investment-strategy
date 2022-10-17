import pandas as pd
import numpy as np
from datetime import datetime,timedelta


class Backtesting():
    """
    Class to Backtest our two strategies
    """
    
    def __init__(self, brokerFees: float):
        self.brokerFees = brokerFees
      
    def getYesterday(self, d):
        d_datetime = datetime.strptime(d,'%d/%m/%Y') - timedelta(days=1)
        return(d_datetime.strftime('%d/%m/%Y'))   
        
    def backtesting1(self,df_returns,week_optimal_allocation):
        """
        Function to backtest Startegy one
        ----------
        **Parameters:
        df_returns : pd.DataFrame
        week_optimal_allocation: dict
        **Returns:
        -------
        tuple : (dict,dict)
        """ 
        
        portfolio_dict = {}
        portfolio_value = {}

        for day in df_returns.index:

            if portfolio_dict == {}:
                portfolio_dict[day] = {
                    week_optimal_allocation[day][0][i]:week_optimal_allocation[day][1][i]*(1-self.brokerFees) 
                    for i in  range(10)
                }
                portfolio_value[day]=1-self.brokerFees
                continue

            else :
                yesterday = self.getYesterday(day)
                portfolio_dict[day] = {defi:value*(1+df_returns.loc[day][defi]) for defi,value in portfolio_dict[yesterday].items()}
                portfolio_value[day] = sum(portfolio_dict[day].values())

            if day in week_optimal_allocation.keys():
                allocution = {week_optimal_allocation[day][0][i]:week_optimal_allocation[day][1][i] for i in range(10)}
                portfolio_cash = 0
                old_portfolio = portfolio_dict[day].keys()
                new_portfolio = allocution.keys()
                eliminated_defi = [defi for defi in old_portfolio if defi not in new_portfolio]
                new_defi = [defi for defi in new_portfolio if defi not in old_portfolio]
                percisting_defi = [defi for defi in new_portfolio if defi in old_portfolio]

                
                

                for defi in percisting_defi:
                    residual = allocution[defi]*portfolio_value[day] - portfolio_dict[day][defi]
                    if residual >= 0:
                        portfolio_dict[day][defi]+=residual*(1-self.brokerFees)
                        portfolio_cash -= residual
                    else:
                        portfolio_dict[day][defi]+=residual
                        portfolio_cash -= residual*(1-self.brokerFees)

                for defi in eliminated_defi:
                    portfolio_cash += portfolio_dict[day][defi]*(1-self.brokerFees)
                    del portfolio_dict[day][defi]

                for defi in new_defi:
                    if portfolio_cash>(allocution[defi]*portfolio_value[day]):
                        portfolio_dict[day][defi] = allocution[defi]*portfolio_value[day]*(1-self.brokerFees)
                        portfolio_cash -= allocution[defi]*portfolio_value[day]
                    else :
                        portfolio_dict[day][defi] = portfolio_cash*(1-self.brokerFees)
                        portfolio_cash -= portfolio_cash
                        break
                        
        return(portfolio_dict,portfolio_value)
    
    def backtesting2(self, df_returns, week_optimal_allocation, equal_weight_index,window=60):
        """
        Function to backtest Startegy two
        ----------
        **Parameters:
        df_returns : pd.DataFrame
        week_optimal_allocation: dict
        equal_weight_index:pd.DataFrame
        **Returns:
        -------
        tuple : (dict,dict)
        """ 

        portfolio_dict = {}
        portfolio_value = {}

        def sma(index_,rolling_window=window):
            sma60 = index_.copy('deep')
            for defi in sma60.columns:
                sma60[defi] = equal_weight_index[defi].rolling(rolling_window).mean()
            sma60.dropna(inplace=True)
            sma60 = pd.concat([equal_weight_index[:rolling_window-1]*0,sma60])
            return(sma60)

        sma60 = sma(equal_weight_index,window)

        for day in df_returns.index:

            yesterday = self.getYesterday(day)

            if portfolio_dict == {}:
                portfolio_dict[day] = {
                    week_optimal_allocation[day][0][i]:week_optimal_allocation[day][1][i]*(1-self.brokerFees) 
                    for i in range(10)
                }
                portfolio_value[day]=1-self.brokerFees
                continue

            else :

                if portfolio_dict[yesterday] !={}:
                    portfolio_dict[day] = {defi:value*(1+df_returns.loc[day][defi]) for defi,value in portfolio_dict[yesterday].items()}
                    portfolio_value[day] = sum(portfolio_dict[day].values())

                else:
                    portfolio_dict[day] = {}
                    portfolio_value[day] = portfolio_value[yesterday]

            if day in week_optimal_allocation.keys():

                if ((equal_weight_index.loc[day]-sma60.loc[day]).values[0] < 0): 

                    if portfolio_dict[day] != {}:
                        portfolio_dict[day] = {}
                        portfolio_value[day] = portfolio_value[day]*(1-self.brokerFees)

                    else:
                        continue

                else:

                    allocution = {week_optimal_allocation[day][0][i]:week_optimal_allocation[day][1][i] for i in range(10)}
                    new_portfolio = allocution.keys()

                    if portfolio_dict[day] == {}:
                        portfolio_cash = portfolio_value[day]
                        for defi in new_portfolio:
                            if portfolio_cash>(allocution[defi]*portfolio_value[day]):
                                portfolio_dict[day][defi] = allocution[defi]*portfolio_value[day]*(1-self.brokerFees)
                                portfolio_cash -= allocution[defi]*portfolio_value[day]
                            else :
                                portfolio_dict[day][defi] = portfolio_cash*(1-self.brokerFees)
                                portfolio_cash -= portfolio_cash
                                break

                    else:
                        portfolio_cash = 0
                        old_portfolio = portfolio_dict[day].keys()
                        eliminated_defi = [defi for defi in old_portfolio if defi not in new_portfolio]
                        new_defi = [defi for defi in new_portfolio if defi not in old_portfolio]
                        percisting_defi = [defi for defi in new_portfolio if defi in old_portfolio]

                        for defi in percisting_defi:
                            residual = allocution[defi]*portfolio_value[day] - portfolio_dict[day][defi]
                            if residual >= 0:
                                portfolio_dict[day][defi]+=residual*(1-self.brokerFees)
                                portfolio_cash -= residual
                            else:
                                portfolio_dict[day][defi]+=residual
                                portfolio_cash -= residual*(1-self.brokerFees)

                        for defi in eliminated_defi:
                            portfolio_cash += portfolio_dict[day][defi]*(1-self.brokerFees)
                            del portfolio_dict[day][defi]

                        for defi in new_defi:
                            if portfolio_cash>(allocution[defi]*portfolio_value[day]):
                                portfolio_dict[day][defi] = allocution[defi]*portfolio_value[day]*(1-self.brokerFees)
                                portfolio_cash -= allocution[defi]*portfolio_value[day]
                            else :
                                portfolio_dict[day][defi] = portfolio_cash*(1-self.brokerFees)
                                portfolio_cash -= portfolio_cash
                                break

        return(portfolio_dict,portfolio_value)

