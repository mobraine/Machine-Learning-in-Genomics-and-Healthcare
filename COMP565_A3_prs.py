import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
ld = pd.read_csv('/Users/Mobrain1/Desktop/COMP 565/A3/data/LD.csv.gz')
ld.set_index("Unnamed: 0", inplace = True)
beta_marginal = pd.read_csv('/Users/Mobrain1/Desktop/COMP 565/A3/data/beta_marginal.csv.gz')
beta_marginal.set_index("Unnamed: 0", inplace = True)

x_test = pd.read_csv('/Users/Mobrain1/Desktop/COMP 565/A3/data/X_test.csv.gz')
x_train = pd.read_csv('/Users/Mobrain1/Desktop/COMP 565/A3/data/X_train.csv.gz')
y_test = pd.read_csv('/Users/Mobrain1/Desktop/COMP 565/A3/data/y_test.csv.gz')
y_train = pd.read_csv('/Users/Mobrain1/Desktop/COMP 565/A3/data/y_train.csv.gz')

M = 100
N_train = 439
N_test = 50

miu_star_beta_all = np.full(100, 0.0)
tau_star_beta_all = np.full(100, 1.0)
gamma_star_all = np.full(100, 0.01)

tau_epis = 1 
tau_beta = 200 
pi = 0.01

#%% Q1 Expectation
def e_step (tau_epis, tau_beta, pi, miu_star_beta_all, tau_star_beta_all, gamma_star_all):
    for j in tqdm.tqdm(range(M)):
        # tau_star_beta_j = N_train * np.diag(ld)[j] * tau_epis + tau_beta # here I assume X'_jX_j is not equal to N_train, because X'_jX_j / N != 1
        tau_star_beta_j = N_train * tau_epis + tau_beta # or I can assume X'_jX_j is equal to N_train, in which case I don't have to multiply by np.diag(ld)
        # tau_star_beta_all[j] = tau_star_beta_j

        def inner_(j, gamma_star_all, miu_star_beta_all):
            beta_marginal_j = beta_marginal.iloc[j].iloc[0]
            sum_no_j = 0
            for i in range(M):
                if i != j:
                    sum_no_j += gamma_star_all[i] * miu_star_beta_all[i] * ld.iloc[j].iloc[i]
            return beta_marginal_j - sum_no_j
        
        # miu_star_beta_j = N_train * tau_epis / tau_star_beta_all[j] * inner_(j, gamma_star_all, miu_star_beta_all)
        miu_star_beta_j = N_train * tau_epis / tau_star_beta_j * inner_(j, gamma_star_all, miu_star_beta_all)
        # miu_star_beta_all[j] = miu_star_beta_j

        # u_j = np.log(pi / (1-pi)) + 0.5 * np.log(tau_beta / tau_star_beta_all[j]) + 0.5 * tau_star_beta_all[j] * np.power(miu_star_beta_all[j], 2)
        u_j = np.log(pi / (1-pi)) + 0.5 * np.log(tau_beta / tau_star_beta_j) + 0.5 * tau_star_beta_j * np.power(miu_star_beta_j, 2)
        gamma_star_j = 1 / (1 + np.exp(-u_j))
        # gamma_star_all[j] = gamma_star_j

        tau_star_beta_all[j] = tau_star_beta_j
        miu_star_beta_all[j] = miu_star_beta_j
        gamma_star_all[j] = gamma_star_j

    # cap gamma_j between 0.01 and 0.99
    gamma_star_all[gamma_star_all < 0.01] = 0.01
    gamma_star_all[gamma_star_all > 0.99] = 0.99
   
    # return gamma_star_all, miu_star_beta_all, tau_star_beta_all

#%% Q2 Maximization
def m_step (gamma_star_all, miu_star_beta_all, tau_star_beta_all):
    pi = 0
    numerator = 0
    denominator = 0
    for j in tqdm.tqdm(range(M)):
        pi += gamma_star_all[j] / M
        numerator += gamma_star_all[j] * (np.power(miu_star_beta_all[j], 2) + 1 / tau_star_beta_all[j])
        denominator += gamma_star_all[j]
    tau_beta = denominator / numerator # tau_beta^(-1) = numerator / denominator, therefore tau_beta = denominator / numerator
    # return tau_beta, pi

#%% Q3 ELBO
def ELBO (tau_epis, tau_beta, pi, miu_star_beta_all, tau_star_beta_all, gamma_star_all):
    # elements_one = [0.5 * tau_epis * (gamma_star_all[j] * (np.power(miu_star_beta_all[j],2) + 1 / tau_star_beta_all[j])) * np.diag(ld)[j] for j in range(M)]
    elements_one = [0.5 * tau_epis * (gamma_star_all[j] * (np.power(miu_star_beta_all[j],2) + 1 / tau_star_beta_all[j])) * N_train for j in range(M)]
    sum_elements_one = np.sum(elements_one)
    # elements_two = [gamma_star_all[j] * miu_star_beta_all[j] * gamma_star_all[k] * miu_star_beta_all[k] * ld.iloc[k].iloc[j] for j in range(M) for k in range(j+1, M)]
    elements_two = [gamma_star_all[j] * miu_star_beta_all[j] * gamma_star_all[k] * miu_star_beta_all[k] * ld.iloc[k].iloc[j] * N_train for j in range(M) for k in range(j+1, M)]
    sum_elements_two = tau_epis * np.sum(elements_two)

    expec_ln_prob_y_beta_s = 0.5 * N_train * np.log(tau_epis) - 0.5 * tau_epis * N_train + \
        np.squeeze(tau_epis * np.matmul(np.multiply(gamma_star_all, miu_star_beta_all), beta_marginal * N_train)) - \
            sum_elements_one - sum_elements_two
    
    elements_three = [-0.5 * np.log(2 * pi * 1 / tau_beta) - tau_beta * 0.5 * gamma_star_all[j] * (np.power(miu_star_beta_all[j],2) + 1 / tau_star_beta_all[j]) for j in range(M)]
    # elements_three = [-0.5 * np.log(2 * pi * 1 / tau_star_beta_all[j]) - tau_beta * 0.5 * gamma_star_all[j] * (np.power(miu_star_beta_all[j],2) + 1 / tau_star_beta_all[j]) for j in range(M)]
    expec_ln_prob_beta_s = np.sum(elements_three)

    elements_four = [-0.5 * np.log(2 * pi * 1 / tau_beta) - 0.5 * gamma_star_all[j] * np.log(tau_beta) for j in range(M)]
    # elements_four = [-0.5 * np.log(2 * pi * 1 / tau_star_beta_all[j]) - 0.5 * gamma_star_all[j] * np.log(tau_beta) for j in range(M)]
    expec_ln_qdist_beta_s = np.sum(elements_four)

    elements_five = [gamma_star_all[j] * np.log(pi) + (1-gamma_star_all[j]) * np.log(1 - pi) for j in range(M)]
    expec_ln_prob_s_pi = np.sum(elements_five)

    elements_six = [gamma_star_all[j] * np.log(gamma_star_all[j]) + (1-gamma_star_all[j]) * np.log(1 - gamma_star_all[j]) for j in range(M)]
    expec_ln_qdist_s = np.sum(elements_six)

    ELBO_VALUE = expec_ln_prob_y_beta_s + expec_ln_prob_beta_s + expec_ln_prob_s_pi - expec_ln_qdist_beta_s - expec_ln_qdist_s

    return ELBO_VALUE

ELBO_VALUES = []

for i in tqdm.tqdm(range(10)):
    # m_step(gamma_star_all, miu_star_beta_all, tau_star_beta_all)
    e_step(tau_epis, tau_beta, pi, miu_star_beta_all, tau_star_beta_all, gamma_star_all)
    m_step(gamma_star_all, miu_star_beta_all, tau_star_beta_all)
    ELBO_VALUES.append(ELBO(tau_epis, tau_beta, pi, miu_star_beta_all, tau_star_beta_all, gamma_star_all))

plt.scatter(x = range(1,11), y = ELBO_VALUES)
plt.show()


#%% Q4
y_train_pred = np.matmul(x_train.drop('Unnamed: 0', axis=1), np.multiply(gamma_star_all, miu_star_beta_all)) 
y_test_pred = np.matmul(x_test.drop('Unnamed: 0', axis=1), np.multiply(gamma_star_all, miu_star_beta_all))
pearson_r_train = np.corrcoef(y_train_pred, y_train['V1'])
pearson_r_test = np.corrcoef(y_test_pred, y_test['V1'])

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(5, 4),sharey=True)
ax[0].scatter(x = y_train_pred, y = y_train['V1'], s=2)
ax[0].set_title("Train (Pearson Correlation Coef. = " + str(pearson_r_train[0][1]) + ")", fontsize=5)
m_train, b_train = np.polyfit(y_train_pred, y_train['V1'], 1)
ax[0].plot(y_train_pred, m_train * y_train_pred + b_train, color ='red')

ax[1].scatter(x = y_test_pred, y = y_test['V1'], s=2)
m_test, b_test = np.polyfit(y_test_pred, y_test['V1'], 1)
ax[1].set_title("Test (Pearson Correlation Coef. = " + str(pearson_r_test[0][1]) + ")", fontsize=5)
ax[1].plot(y_test_pred, m_test * y_test_pred + b_test, color ='red')

fig.supxlabel("Predicted phenotype", fontsize=8)
fig.supylabel("True phenotype", fontsize=8)
plt.show()

#%% Q5
SNPs = list(beta_marginal.index)
causal_inds = [SNPs.index('rs9482449'), SNPs.index('rs7771989'), SNPs.index('rs2169092')]
plt.scatter(x=range(100), y=gamma_star_all)
plt.scatter(x=causal_inds, y=gamma_star_all[causal_inds])
plt.ylabel('PIP')
plt.xlabel('SNP')
plt.show()