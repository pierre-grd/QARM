# QARM
Quantitative Assets and Risk management 


stock_mu = pct_change_mean.shape[0]
e = np.ones((stock_mu, 1))
pct_change_mean =np.array(pct_change_mean)
cov_excess = np.array(cov_excess)
covin = np.linalg.inv(cov_excess)
def MVPs(lambd, stocks, cov):
    cov_in = np.linalg.inv(cov)
    stock_mu = stocks.shape[0]
    e = np.ones((stock_mu, 1))
    A = (cov_in @ e)/(e.T @ cov_in @ e)
    B = (1/lambd) * cov_in
    C = ((e.T @ cov_in @ stocks)/(e.T @ cov_in @ e))*e
    D = stocks - C
    alpha = A + B@D
    return alpha

print(MVPs(5, pct_change_mean, cov_excess))
lambdas = list(np.linspace(5,1000,100))

portfolio_returns = []

portfolio_volatilities = []

for ra in lambdas:
    weights = MVPs(ra, pct_change_mean, cov_excess)
    retur_n = weights.T @ pct_change_mean
    volat = np.sqrt(weights.T @ cov_excess @ weights)
    portfolio_returns.append(retur_n)
    portfolio_volatilities.append(volat)


print((covin@e)/e.T@covin@e)