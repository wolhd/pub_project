#%%
import pandas as pd 
import numpy as np
#%%
# scratch
a = np.arange(1,13).reshape((3,4),order='F')
# %%
tdf = pd.read_csv('data_walmart_train.csv')
tmdf = pd.read_csv('data_walmart_train_missing.csv')

# %%
print(f'tdf {tdf.isnull().sum().sum()}')
print(f'tmdf {tmdf.isnull().sum().sum()}')

# %%
# Compute the Page matrix of two stores, s10_d83 and s1_d34
s10 = tdf['s10_d83'].to_numpy()
s1 = tdf['s1_d34'].to_numpy()
L = 10
N = len(s10)
n_col = int(N/L) # num cols
pg_s10 = s10.reshape((L,n_col),order='F')
pg_s1 = s1.reshape((L,n_col),order='F')
#%%
# missing data version
ms10 = tmdf['s10_d83'].to_numpy()
ms1 = tmdf['s1_d34'].to_numpy()
zs10 = np.nan_to_num(ms10)
zs1 = np.nan_to_num(ms1)
L = 10
N = len(zs10)
n_col = int(N/L) # num cols
pg_zs10 = zs10.reshape((L,n_col),order='F')
pg_zs1 = zs1.reshape((L,n_col),order='F')
#%%
def compute_r(page_mat):
    u,s,vh = np.linalg.svd(page_mat)
    s_sq = s**2
    s_sq_fr = [ ss / s_sq.sum() for ss in s_sq]
    print(f's sq fract {s_sq_fr}')
    print(f's sq fr cumsum {np.cumsum(s_sq_fr)}')
#%%
print(f'store 10')
compute_r(pg_s10)
print(f'store 1')
compute_r(pg_s1)

# %%
print(f'missing data store10')
compute_r(pg_zs10)
print(f'missing data store1')
compute_r(pg_zs1)

#%%
# Imputation, calc singular values on missing data page matrix,
#  normalize/adjust because of zeros for missing values
#  Recompute page matrix with only first r singular values
#  Normalize new page matrix by 1/p, p = proportion of obs vals (T-n)/T
#  T total len, n num missing vals
in_mis_arr = ms10 # input data array with missing vals
orig_pg_mat = pg_s10 # original, no missing vals

# in_mis_arr = ms1.copy() # input data array with missing vals
# orig_pg_mat = pg_s1.copy() # original, no missing vals

r = 5
T = len(in_mis_arr)
n = np.isnan(in_mis_arr).sum()
p = (T-n)/T
L = 10
n_col = int(T/L) # num cols
mis_pg_mat = in_mis_arr.reshape((L,n_col),order='F').copy()
mis_inds = np.isnan(mis_pg_mat)
z_pg_mat = mis_pg_mat
z_pg_mat[mis_inds] = 0
u,s,vh = np.linalg.svd(z_pg_mat)
n_s = len(s)
S = np.zeros(z_pg_mat.shape) # create s with only r singular vals
s_r = s.copy()
s_r[r:] = 0
S[:n_s, :n_s] = np.diag(s_r)
new_pg_mat = u@S@vh
new_pg_mat = 1/p * new_pg_mat
sqerr = (new_pg_mat - orig_pg_mat)**2
impute_sqerrs = sqerr[mis_inds]
mse = np.mean(impute_sqerrs)
print(f'mse {mse}')
#%%
mse_s1 = mse
# %%

------
4 autocov
3 hankel
8 short term predict
--
15

5 multiple time ser

4 page matrix
2 imputation
2 influential factors
--
8

6 multivariate ssa
2 mv impacts
--
8

3 cricket match scores
2 synthetic control
2 weights
1 early
--
8


29+15=44



