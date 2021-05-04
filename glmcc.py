#Author: Junichi Haruna
# coding: utf-8
"""
Functions for fitting a GLM to the Cross correlation.
Refer to docstring for explanation of each function.
In addition, the following functions can be used from the command.

* make cross correlogram
* output a result by fitting GLM to cross correlation
* output a result by fitting GLM using C and Python
* output a result by fitting GLM using C and Python and only exp(a_k) result

Refference:
Reconstructing neuronal circuitry from parallel spike train.

-------------------------------------------------------------------------------
Cross correlogramをGLMでフィッティングする関数を書いています。
関数の説明はdocstringを参照してください。
なお、ターミナル(Windowsならコマンドプロンプト)から以下の機能をコマンド入力により実行できます。

* Cross correlogramの図を作成する
* Cross correlogramの上にGLMでフィッティングした結果を出力する
* PythonとC言語によりGLMでフィッティングした結果を重ね書きする。
* PythonとC言語によりGLMでフィッティングした結果をexp(a_k)と共に重ね書きする

-------------------------------------------------------------------------------
変数の設定
linear_corssCorrelogram:
cell1: セルデータ1(list)
cell2: セルデータ2(list)

bin_width: ヒストグラムの棒の幅
bin_num: ヒストグラムの棒の数
hist_array: ヒストグラムの結果をnumpyの配列にいれたもの

calc_hessian:
par: 予測するパラメータ
hessian: numpyの行列、求めたヘッセ行列を入れる

calc_grad_log_p:
g_log_p: 対数事後確率の勾配をnumpyの配列にいれたもの

LM:
Gk: Gkを入れるlist
log_pos: 対数事後確率をlistにいれたもの
grad: 対数事後確率の勾配をnumpyの縦ベクトルにいれたもの
hessian: 対数事後確率のヘッセ行列
new_par: 更新したパラメータ

divide_into_E_I:
W: PSPの行列(list)
firing_rate: 発火率(list)
sorted_firing_rate: 昇順に直した発火率
W_e_t: Wの興奮性だけの集合(発火率が昇順)
W_i_t: Wの抑制性だけの集合(発火率が昇順)
e_cell_list: 興奮性のセルの集合
i_cell_list: 抑制性のセルの集合

plt_W_GLMCC_with_A:
bin_width: ヒストグラムの棒の幅
bin_num: ヒストグラムの棒の数
fit_model: フィッティングした結果(list)
a_model: exp(a(k))だけのリスト(list) (J_+, J_-の情報なし)
hist_array: ヒストグラムのデータ(listにしている)

"""

import matplotlib.pyplot as plt
import scipy.special as special
import numpy as np
import math
import sys
import time
import subprocess
import csv

WIN = 50.0
DELTA = 1.0
NPAR = 102
MAX = 200000

def index_linear_search(list, target, index):
    '''
    search for index with the smallest value which is bigger than target.
    ---------------------------------------------------------------------
    targetよりも大きくて、その中でも一番小さい値のindexを返す関数
    '''
    
    result = 0
    if index == -1:
        while len(list) > result and list[result] <= target:
            result += 1
        return result
    else:
        result = index
        while len(list) > result and list[result] <= target:
            result += 1
        return result

def linear_crossCorrelogram(filename1, filename2, T):
    '''
    make Cross correlogram.
    
    Input:
    file name 1, file name 2, T(s)(float or int)

    Output:
    list of spike time (list)
    list of histogram (list)
    the number of cell1's spike time (list)
    the number of cell2's spike time (list)

    -------------------------------------------
    Cross correlogramの図を作成する。

    入力:
    ファイルの名前1, ファイルの名前2, T(s)

    出力:
    スパイク時間のリスト
    ヒストグラムのリスト
    cell1のスパイク時間の数
    cell2のスパイク時間の数
    
    '''

    #open cell data
    cell_file1 = open(filename1, "r")
    cell_file2 = open(filename2, "r")

    cell1 = cell_file1.readlines()
    cell2 = cell_file2.readlines()

    for i in range(len(cell1)-1, -1, -1):
        cell1[i] = float(cell1[i])
        if cell1[i] >= T*1000.0 or cell1[i] < 0:
            del cell1[i]

    for i in range(len(cell2)-1, -1, -1):
        cell2[i] = float(cell2[i])
        if cell2[i] >= T*1000.0 or cell2[i] < 0:
            del cell2[i]

    cell_file1.close()
    cell_file2.close()

    print('n_pre: '+str(len(cell1)))
    print('n_post: '+str(len(cell2)))

    # make c_ij(spike time)
    w = int(WIN)
    c = []
    min_index = -1
    max_index = -1

    for i in range(len(cell2)):
        min = cell2[i] - w
        max = cell2[i] + w

        min_j = index_linear_search(cell1, min, min_index)
        min_index = min_j
        max_j = index_linear_search(cell1, max, max_index)
        max_index = max_j

        c_i = []
        for j in range(max_j - min_j):
            if (cell1[min_j + j] - cell2[i]) < WIN:
                c_i.append(cell1[min_j + j] - cell2[i])

        c.extend(c_i)

    
    # make histogram
    bin_width = DELTA # bin width
    bin_num = int(2 * w / bin_width) # the number of bin
    
    hist_array = np.histogram(np.array(c), bins=bin_num, range=(-1*w, w))
    result = [0, 0, 0, 0]
    result[0] = c
    result[1] = hist_array[0].tolist()
    result[2] = len(cell1)
    result[3] = len(cell2)

    return result

"""

def plt_CrossCorrelogram(filename1, filename2, T):
    '''
    make Cross correlogram.
    
    Input:
    file name 1, file name 2, T(s)(float or int)

    Output:
    list of spike time (list)
    list of histogram (list)
    the number of cell1's spike time (list)
    the number of cell2's spike time (list)

    -------------------------------------------
    Cross correlogramの図を作成する。

    入力:
    ファイルの名前1, ファイルの名前2, T(s)

    出力:
    スパイク時間のリスト
    ヒストグラムのリスト
    cell1のスパイク時間の数
    cell2のスパイク時間の数
    
    '''

    #open cell data
    cell_file1 = open(filename1, "r")
    cell_file2 = open(filename2, "r")

    cell1 = cell_file1.readlines()
    cell2 = cell_file2.readlines()

    for i in range(len(cell1)-1, -1, -1):
        cell1[i] = float(cell1[i])
        if cell1[i] >= T*1000.0 or cell1[i] < 0:
            del cell1[i]

    for i in range(len(cell2)-1, -1, -1):
        cell2[i] = float(cell2[i])
        if cell2[i] >= T*1000.0 or cell2[i] < 0:
            del cell2[i]

    cell_file1.close()
    cell_file2.close()

    print('n_pre: '+str(len(cell1)))
    print('n_post: '+str(len(cell2)))

    # make c_ij(spike time)
    w = int(WIN)
    c = []
    min_index = -1
    max_index = -1

    for i in range(len(cell2)):
        min = cell2[i] - w
        max = cell2[i] + w

        min_j = index_linear_search(cell1, min, min_index)
        min_index = min_j
        max_j = index_linear_search(cell1, max, max_index)
        max_index = max_j

        c_i = []
        for j in range(max_j - min_j):
            if (cell1[min_j + j] - cell2[i]) < WIN:
                c_i.append(cell1[min_j + j] - cell2[i])

        c.extend(c_i)

    
    # make histogram
    bin_width = DELTA # bin width
    bin_num = int(2 * w / bin_width) # the number of bin
    
    plt.hist(np.array(c), bins=bin_num, range=(-1*w, w))
    plt.xlim(-50, 50)
    #result = [0, 0, 0, 0]
    #result[0] = c
    #result[1] = hist_array[0].tolist()
    #result[2] = len(cell1)
    #result[3] = len(cell2)

    #return result

"""

def init_par(rate):
    '''
    initialize parameter

    ---------------------
    パラメータを初期化する
    '''
    par = np.ones((NPAR, 1))
    par = math.log(rate)*par
    par[NPAR-2][0] = 0.1
    par[NPAR-1][0] = 0.1
    return par


def calc_hessian(par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk):
    '''
    calculate hessian of log posterior probability

    Output:
    numpy matrix

    -------------------------------------------------
    対数事後確率のヘシアンを計算する
    
    出力:
    numpyの行列
    '''
    hessian = np.zeros((NPAR, NPAR))

    #d^2P/da_kda_l, d^2P/da_kdJ
    for i in range(0, NPAR-2):
        for j in range(0, NPAR):
            #d^2P/da_kdJ
            if j == NPAR-2:
                x_k = (i + 1)*DELTA - WIN
                if x_k > delay_synapse:
                    # if abs(J) < 1.0e-3, approximate J=0
                    if abs(par[NPAR-2][0]) < 1.0e-3:
                        hessian[i][j] = tau[0]*np.exp(par[i][0])*func_f(x_k-DELTA, delay_synapse, tau[0])*(1-np.exp(-DELTA/tau[0]))

                    else:
                        hessian[i][j] = (-1)*(tau[0] * np.exp(par[i][0])/par[NPAR-2][0]) * (np.exp(par[NPAR-2][0] * func_f(x_k-DELTA, delay_synapse, tau[0])) - np.exp(par[NPAR-2][0] * func_f(x_k, delay_synapse, tau[0])))
                
                    
            elif j == NPAR-1:
                x_k = (i + 1)*DELTA - WIN
                if x_k <= (-1) * delay_synapse:
                    # if abs(J) < 1.0e-3, approximate J=0
                    if abs(par[NPAR-1][0]) < 1.0e-3:
                        hessian[i][j] = tau[1]*np.exp(par[i][0])*func_f(-x_k, delay_synapse, tau[1])*(1-np.exp(-DELTA/tau[1]))
                        
                    else:
                        hessian[i][j] = (-1)*(tau[1] * np.exp(par[i][0])/par[NPAR-1][0]) * (np.exp(par[NPAR-1][0] * func_f(-x_k, delay_synapse, tau[1])) - np.exp(par[NPAR-1][0] * func_f(-x_k+DELTA, delay_synapse, tau[1])))

            #d^2p/da_kda_l
            else:
                if i == j:
                    hessian[i][j] = (-1) * Gk[i] + (beta/DELTA)*(K_delta(i, 0) + K_delta(i, NPAR-3) - 2)
                else:
                    hessian[i][j] = (beta/DELTA)*(K_delta(i-1, j) + K_delta(i+1, j))
            
    #d^2P/dJ^2
    for i in range(NPAR-2, NPAR):
        for j in range(0, NPAR):
            if j >= NPAR-2:
                if i == j == NPAR-2:
                    tmp = 0
                    for k in range(0, NPAR-2):
                        x_k = (k + 1) * DELTA - WIN
                        if x_k >delay_synapse:
                            # if abs(J) < 1.0e-3, approximate J=0
                            if abs(par[NPAR-2][0]) < 1.0e-3:
                                tmp = (tau[0]/2)*(func_f(x_k-DELTA, delay_synapse ,tau[0])**2)*(1-np.exp(-2*DELTA/tau[0]))
                                hessian[i][j] -= tmp
                            else:
                                tmp = (par[NPAR-2][0]*func_f(x_k-DELTA, delay_synapse, tau[0]) - 1) * np.exp(par[NPAR-2][0]*func_f(x_k-DELTA, delay_synapse, tau[0]))
                                tmp -= (par[NPAR-2][0]*func_f(x_k, delay_synapse, tau[0]) -1) * np.exp(par[NPAR-2][0]*func_f(x_k, delay_synapse, tau[0]))
                                hessian[i][j] -= ((tau[0]*np.exp(par[k][0]))/(par[NPAR-2][0]**2))*tmp

                elif i == j == NPAR-1:
                    tmp = 0
                    for k in range(0, NPAR-2):
                        x_k = (k + 1) * DELTA - WIN
                        if x_k <= -delay_synapse:
                            # if abs(J) < 1.0e-3, approximate J=0
                            if abs(par[NPAR-1][0]) < 1.0e-3:
                                tmp = (tau[1]/2)*(func_f(-x_k, delay_synapse, tau[1])**2)*(1-np.exp(-2*DELTA/tau[1]))
                                hessian[i][j] -= tmp
                            else:
                                tmp = (par[NPAR-1][0]*func_f(-x_k, delay_synapse, tau[1])-1) * np.exp(par[NPAR-1][0]*func_f(-x_k, delay_synapse, tau[1]))
                                tmp -= (par[NPAR-1][0]*func_f(-x_k+DELTA, delay_synapse, tau[1])-1) * np.exp(par[NPAR-1][0]*func_f(-x_k+DELTA, delay_synapse, tau[1]))
                                hessian[i][j] -= ((tau[1]*np.exp(par[k][0]))/(par[NPAR-1][0]**2))*tmp 
                    
            else:
                hessian[i][j] = hessian[j][i]
    
                    
    return hessian

def calc_grad_log_p(par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk):
    '''
    calculate gradient of log posterior probability

    Output:
    numpy column vector

    -----------------------------------------------------
    対数事後確率の勾配を計算する

    出力:
    numpyの列ベクトル
    '''
    g_log_p = np.zeros((NPAR, 1))

    #dP/da_k
    
    for i in range(0, NPAR-2):
        tmp = 0
        g_log_p[i][0] = -Gk[i]
        
        if i == 0:
            tmp = (-1) * (par[i][0] - par[i+1][0])
        elif i == NPAR-3:
            tmp = (-1) * (par[i][0] - par[i-1][0])
        else:
            tmp = (-1) * (par[i][0] - par[i-1][0]) + (-1) * (par[i][0] - par[i+1][0])
        g_log_p[i][0] += (beta/DELTA)*tmp + c[i]

    #dP/dJ_ij, dp/dJ_ji
    tmp_ij = 0
    tmp_ji = 0

    for i in range(0, n_sp):
        '''
        #実験データの時使用する
        if 1 < abs(t_sp[i]):
        
            if t_sp[i] > delay_synapse:
                tmp_ij += func_f(t_sp[i], delay_synapse, tau[0])

            elif t_sp[i] < -delay_synapse:
                tmp_ji += func_f(-t_sp[i], delay_synapse, tau[1])
        '''
        if t_sp[i] > delay_synapse:
            tmp_ij += func_f(t_sp[i], delay_synapse, tau[0])

        elif t_sp[i] < -delay_synapse:
            tmp_ji += func_f(-t_sp[i], delay_synapse, tau[1])


    for i in range(0, NPAR-2):
        x_k = (i + 1)*DELTA - WIN
        # if abs(J) < 1.0e-3, approximate J=0
        if x_k > delay_synapse:
            if abs(par[NPAR-2][0]) < 1.0e-3:
                tmp_ij -=tau[0]*np.exp(par[i][0])*func_f(x_k-DELTA, delay_synapse, tau[0])*(1-np.exp(-DELTA/tau[0]))
            else:
                tmp_ij -= (tau[0] * np.exp(par[i][0])/par[NPAR-2][0]) * (np.exp(par[NPAR-2][0] * func_f(x_k-DELTA, delay_synapse,tau[0])) - np.exp(par[NPAR-2][0] * func_f(x_k, delay_synapse, tau[0])))
        elif x_k <= (-1) * delay_synapse:
            if abs(par[NPAR-1][0]) < 1.0e-3:
                tmp_ji -=tau[1]*np.exp(par[i][0])*func_f(-x_k, delay_synapse, tau[1])*(1-np.exp(-DELTA/tau[1]))
            else:
                tmp_ji -= (tau[1] * np.exp(par[i][0])/par[NPAR-1][0]) * (np.exp(par[NPAR-1][0] * func_f(-x_k, delay_synapse, tau[1])) - np.exp(par[NPAR-1][0] * func_f(-x_k+DELTA, delay_synapse, tau[1])))

    g_log_p[NPAR-2][0] = tmp_ij
    g_log_p[NPAR-1][0] = tmp_ji

    return g_log_p

def calc_Gk(par, beta, tau, c, n_sp, t_sp, delay_synapse):
    '''
    calculate Gk using scipy.special.expi()
    
    Output: Gk (list) 

    -------------------------------------------
    scipyのモジュールを使ってGkを計算する
    
    出力: Gk (list)
    '''
    Gk = [0 for i in range(0, NPAR-2)]
    for i in range(0, NPAR-2):
        x_k = (i+1)*DELTA-WIN
        tmp = 0

        '''
        #実験データの時に使用する
        if i == int(WIN-1) or i == int(WIN):
            continue
        '''
        
        if x_k <= -delay_synapse and abs(par[NPAR-1][0]*func_f(-x_k, delay_synapse, tau[1])) > 1.0e-6:
            tmp = special.expi(par[NPAR-1][0]*func_f(-x_k, delay_synapse, tau[1]))
            tmp -= special.expi(par[NPAR-1][0]*func_f(-x_k+DELTA, delay_synapse, tau[1]))
            
            Gk[i] = tmp*np.exp(par[i][0])*tau[1]
        elif x_k > delay_synapse and abs(par[NPAR-2][0]*func_f(x_k, delay_synapse, tau[0])) > 1.0e-6:
            tmp = special.expi(par[NPAR-2][0]*func_f(x_k-DELTA, delay_synapse, tau[0]))
            tmp -= special.expi(par[NPAR-2][0]*func_f(x_k, delay_synapse, tau[0]))
            
            Gk[i] = tmp*np.exp(par[i][0])*tau[0]
        else:
            Gk[i] = DELTA*np.exp(par[i][0])

    return Gk
            
def K_delta(i, j):
    '''
    Kronecker delta

    ---------------------
    クロネッカーのデルタ
    '''
    if i == j:
        return 1
    else:
        return 0

def func_f(sec, delay, tau):
    '''
    The time profile of the synaptic interaction 
    
    ----------------------------------------------
    シナプス電流の効果を表す関数

    '''
    if sec >= delay:
        return np.exp(-(sec-delay)/tau)
    else:
        return 0  

def calc_log_posterior(par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk):
    '''
    calculate log posterior probability
    Output: log_post (float)

    --------------------------------------
    対数事後確率を求める関数
    出力: log_post(float), log_likelihood(float)
    '''
    log_likelihood = 0
    
    for i in range(0, NPAR):
        log_likelihood += (par[i][0]*c[i])

    for i in range(0, NPAR-2):
        log_likelihood = log_likelihood - Gk[i]
    
    tmp = 0
    for i in range(0, NPAR-3):
        tmp += (par[i+1][0]-par[i][0])**2
    tmp = (beta/(2*DELTA)) * tmp
    
    log_post = log_likelihood - tmp

    return log_post, log_likelihood
    
def LM(par, beta, tau, c, n_sp, t_sp, delay_synapse, cond):
    '''
    calculate the best parameter whose log posterior probability is biggest by LM method
    This function does not end until the termination condition is satisfied.
    
    Output:
    parameter (list), log_post (float), log_likelihood(float) (if LM method's convergence condition is satisfied.)
    or
    false (if loop count is 1000.)

    ----------------------------------------------------------------------------------------
    対数事後確率が一番大きいパラメータをLM法で計算する関数。
    この関数は収束条件を満たさない限り終了しない。(繰り返し回数を1000回超えたら強制的に終了する。)

    '''
    
    C_lm = 0.01
    eta = 0.1
    l_c = 0

    if cond > 0:
        par[NPAR - 3 + cond] = 0
        
    while True:
        l_c += 1
        # Update parameters
        #print("現在のパラメータ")
        #print(par)
        Gk = calc_Gk(par, beta, tau, c, n_sp, t_sp, delay_synapse)
        log_pos, log_likelihood = calc_log_posterior(par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk)
        grad = calc_grad_log_p(par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk)
        hessian = calc_hessian(par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk)
        if cond > 0:
            idx = NPAR - 3 + cond
            grad = np.delete(grad, idx, 0)
            hessian = np.delete(np.delete(hessian, idx, 0), idx, 1)
        h_diag = np.diag(hessian)
        tmp = np.eye(h_diag.shape[0], h_diag.shape[0])
        for i in range(0, tmp.shape[0]):
            tmp[i][i] = h_diag[i]
        if cond > 0:
            idx = NPAR - 3 + cond
            new_par = np.delete(par, idx, axis = 0) - np.dot(np.linalg.inv(hessian + C_lm * tmp), grad)
            new_par = np.insert(new_par, idx, np.zeros(1), axis = 0)
        else:
            new_par = par - np.dot(np.linalg.inv(hessian + C_lm * tmp), grad)

        # Adjust J
        p_min = -3
        p_max = 5
        for i in range(2):
            if new_par[NPAR-2+i] < p_min:
                new_par[NPAR-2+i] = p_min
            if p_max < new_par[NPAR-2+i]:
                new_par[NPAR-2+i] = p_max
        

        # Whether log posterior probability is increasing
        Gk = calc_Gk(new_par, beta, tau, c, n_sp, t_sp, delay_synapse)
        new_log_pos, new_log_likelihood = calc_log_posterior(new_par, beta, tau, c, n_sp, t_sp, delay_synapse, Gk)
        if (new_log_pos >= log_pos):
            par = new_par
            C_lm = C_lm * eta
        else:
            C_lm = C_lm * (1 / eta)
            continue

        # Whether the convergence condition is satisfied
        if (abs(new_log_pos - log_pos) < 1.0e-4):
            return (par.T).tolist()[0], new_log_pos, new_log_likelihood
    
        if l_c > 1000:
            return False
        

def GLMCC(c, t_sp, tau, beta, pre, post, delay_synapse, cond = 0):
    '''
    fit a GLM to the Cross correlogram

    Input:
    Cross correlogram(list), spike time(list), tau(list), beta(float), the number of cell1's spike time(int), the number of cell2's spike time(int), delay_synapse

    Output:
    parameter (list), log_post (float) (if LM method's convergence condition is satisfied.)
    or
    false (if func LM returns false)

    ----------------------------------------------------------------------------------------
    Cross correlogramをGLMでフィッティングする関数。

    '''

    new_c = [0 for i in range(NPAR)]

    for i in range(len(t_sp)):
        '''
        if 1 < abs(t_sp[i]):
            k = (t_sp[i]+WIN)/DELTA
            tmp = math.floor(k)
            if k-tmp == 0:
                new_c[tmp] = new_c[tmp] + 0.5
                new_c[tmp-1] = new_c[tmp-1] + 0.5
            elif 0 <= tmp and tmp < NPAR-2:
                new_c[tmp] += 1
            else:
                print('Error: '+str(t_sp[i]))
                
            if delay_synapse < t_sp[i]:
                new_c[NPAR-2] += np.exp((-1)*(t_sp[i]-delay_synapse)/tau[0])
            if t_sp[i] < -delay_synapse:
                new_c[NPAR-1] += np.exp((t_sp[i]+delay_synapse)/tau[1])
        '''
        k = (t_sp[i]+WIN)/DELTA
        tmp = math.floor(k)
        if k-tmp == 0:
            new_c[tmp] = new_c[tmp] + 0.5
            new_c[tmp-1] = new_c[tmp-1] + 0.5
        elif 0 <= tmp and tmp < NPAR-2:
            new_c[tmp] += 1
        else:
            print('Error: '+str(t_sp[i]))
                
        if delay_synapse < t_sp[i]:
            new_c[NPAR-2] += np.exp((-1)*(t_sp[i]-delay_synapse)/tau[0])
        if t_sp[i] < -delay_synapse:
            new_c[NPAR-1] += np.exp((t_sp[i]+delay_synapse)/tau[1])

    
    #print(new_c)
    rate = len(t_sp)/(2*WIN)
    n_sp = len(t_sp)
    n_pre = pre
    n_post = post
    # Make a prior distribution
    par = init_par(rate)
    #print("rate: "+str(rate))
    #print("par: ", par)
    #print(t_sp)

    return LM(par, beta, tau, new_c, n_sp, t_sp, delay_synapse, cond)

def calc_PSP_LR(J, D, z_a, c_E=2.532, c_I=0.612):
    PSP = 0
    if 2 * D > z_a:
        if J >= 0:
            PSP = c_E * J
        else:
            PSP = c_I * J
    return PSP

def calc_PSP(J, Jth, c_E=2.532, c_I=0.612):
    '''
    calculate PSP

    Output: PSP (float)

    --------------------
    PSPを計算する関数

    出力: PSP (float)
    '''
    PSP = 0
    if J > Jth:
        PSP = J*c_E
    if J < -Jth:
        PSP = J*c_I

    return PSP

def plot_3d(W_file, Plot_f, n):
    '''
    make plot file to make W image.
    
    Input:
    W file name, Plot file name, n(the number of cell)

    ----------------------------------------------------
    Wの図を作るためのプロットファイルを作成する
    Input:
    Wのファイル名, プロットファイル名, n(セルの数)
    '''
    
    #read W_file and make plot file
    plt_f = open(Plot_f, 'w')
    with open(W_file, 'r') as f:
        reader = csv.reader(f)
        
        #row is row vector of W matrix
        i = 0
        for row in reader:
            #Set the value range from -2 to 2
            for j in range(0, n):
                if float(row[j]) > 2:
                    row[j] = '2'
                if float(row[j]) < -2:
                    row[j] = '-2'

            #write the value
            for j in range(0, n):
                plt_f.write(str(i)+'  '+str(j)+'  '+row[j]+'\n')
                plt_f.write(str(i)+'  '+str(j+1)+'  '+row[j]+'\n')
            plt_f.write('\n')

            for j in range(0, n):
                plt_f.write(str(i+1)+'  '+str(j)+'  '+row[j]+'\n')
                plt_f.write(str(i+1)+'  '+str(j+1)+'  '+row[j]+'\n')
            plt_f.write('\n')
            
            i += 1

    plt_f.close()

def divide_into_E_I(W_file, n, cell_dir):
    '''
    sort cell data from least to most firing rate
    and
    divide W into excitatory and inhibitory connections.
    
    Input:
    W file name, n(the number of cell), cell directory name

    ----------------------------------------------------------
    セルデータを発火率の低い順に並べ替え、興奮性と抑制性の二つに分ける関数。

    入力:
    Wのファイル名, n(セルの数), セルデータの入っているディレクトリ名
    
    '''
    W = [[0 for i in range(n)] for j in range(n)]
    # read W file
    with open(W_file, 'r') as f:
        reader = csv.reader(f)

        i = 0
        for row in reader:
            for j in range(0, n):
                W[i][j] = float(row[j])
            i += 1

    firing_rate = []
    for i in range(0, n):
        cell_file = open(cell_dir+"/cell"+str(i)+".txt", "r")
        cell = cell_file.readlines()

        firing_rate.append(len(cell))

    sorted_firing_rate = sorted(firing_rate)

    for i in range(0, n):
        if sorted_firing_rate[i] != firing_rate[i]:
            for j in range(i, n):
                if sorted_firing_rate[i] == firing_rate[j]:
                    tmp = j
                    break
                
            tmp_i = []
            for j in range(0, n):
                tmp_i.append(W[i][j])      
            for j in range(0, n):
                W[i][j] = W[tmp][j]
            for j in range(0, n):
                W[tmp][j] = tmp_i[j]

            for j in range(0, n):
                tmp_i[j] = W[j][i]
            for j in range(0, n):
                W[j][i] = W[j][tmp]
            for j in range(0, n):
                W[j][tmp] = tmp_i[j]

    
    # transport W list
    W_t = [list(x) for x in zip(*W)]
    
    W_e_t = []
    W_i_t = []
    e_cell_list = []
    i_cell_list = []
    cell_list = []
    
    e_i_rate = [0 for i in range(0, n)]
    for i in range(0, n):
        i_rate = 0
        e_rate = 0
        for j in range(0, n):
            if W_t[i][j] > 0:
                e_rate += 1
            elif W_t[i][j] < 0:
                i_rate += 1

        if (e_rate - i_rate) < 0:
            W_i_t.append(W_t[i])
            i_cell_list.append(i)
        else:
            W_e_t.append(W_t[i])
            e_cell_list.append(i)

    W_d_t = W_e_t + W_i_t
    cell_list = e_cell_list + i_cell_list

    W = [list(x) for x in zip(*W_d_t)]

    for i in range(0, n):
        if cell_list[i] != i:
            for j in range(i, n):
                if cell_list[j] == i:
                    tmp = j
                    break

            tmp_i = []
            for j in range(0, n):
                tmp_i.append(W[i][j])

            for j in range(0, n):
                W[i][j] = W[tmp][j]
            for j in range(0, n):
                W[tmp][j] = tmp_i[j]
                
    W_d_f = open("sorted_W.csv", 'w')

    for i in range(0, n):
        for j in range(0, n):
            if W[i][j] == 0:
                W[i][j] = int(W[i][j])
            W_d_f.write(str(W[i][j]))
            if j == n-1:
                W_d_f.write('\n')
            else:
                W_d_f.write(', ')
                
    W_d_f.close()


def plot_PSP_est(Plot_f, T, n):
    '''
    make W 3d plot.
    
    Input:
    Plot file name, T, n
    
    use gnuplot.
    Output is a planar representation of 3D mapping.(eps file)
    The color is represented by red and blue gradation.

    ------------------------------------------------------------
    Wの3Dプロットの図を作る関数。
    Gnuplotを使用しています。

    入力:
    プロットのファイル名, T, n(セルデータの数)
    '''

    # use gnuplot
    gnuplotcmd ="""
    set ter post eps color enhanced;
    set out "PSP_{0}.eps";
    set size ratio 1;
    set pm3d map;
    set cbrange[-2:2];
    show tics;
    set xl \"post\";
    set yl \"pre\";
    set yrange[{1}:0];
    set palette define (-2 "blue", 0 "white", 2 "red");
    splot [0:{1}][0:{1}] "plt.txt" u 1:2:3;
    """.format(T, n)
    subprocess.call( [ 'gnuplot', '-e', gnuplotcmd])
                    
def plt_GLMCC(c, par, tau, delay_synapse, label_name):
    '''
    make histogram and GLMCC using matplotlib.

    If you want to show graph, run "plt.show()" on Python.
    
    Input:
    Cross correlogram(list), parameter(list), tau(list), delay_synapse, label_name(C or Python)
    
    ----------------------------------------------------------------------------------------
    matplotlibを使ってヒストグラムとGLMCCを描いた図を作成する関数。
    
    入力:
    Cross correlogram(list), パラメータ(list), tau(list), delay_synapse, ラベル名(C or Python)

    '''
    # make histogram
    bin_width = 1 # bin width
    bin_num = int(2 * WIN / bin_width) # the number of bin
    print(len(par))
    print(NPAR-2)
    fit_model = [math.exp(par[i]+par[NPAR-2]*func_f((i+1)*DELTA-WIN, delay_synapse, tau[0])+par[NPAR-1]*func_f(-(i+1)*DELTA+WIN, delay_synapse, tau[1])) for i in range(0, NPAR-2)]
    x = [i for i in range(-int(WIN-1), int(WIN+1))]

    fit_model.insert(int(WIN-1)+delay_synapse, fit_model[int(WIN-1)+delay_synapse-1])
    fit_model.insert(int(WIN-1)-delay_synapse+1, fit_model[int(WIN-1)-delay_synapse+1])
    x = [i for i in range(-int(WIN-1), int(WIN+1))]
    x.insert(int(WIN-1)+delay_synapse, delay_synapse)
    x.insert(int(WIN-1)-delay_synapse+1, -delay_synapse)
    
    Pycolorlist = ['blue', 'cyan']
    Ccolorlist = ['red', 'rosybrown']

    if label_name == 'Py':
        
        #plt.plot([i for i in range(-int(WIN-1), int(WIN+1))], fit_model, linewidth=2, color=Pycolorlist[0],label='GLMCC fitting('+label_name+')')
        plt.plot(x, fit_model, linewidth=1, color=Pycolorlist[0],label='GLMCC fitting('+label_name+')')
        plt.hist(c, bins=bin_num, range=(-1 * WIN, WIN), color=Pycolorlist[1], label='Cross correlogram('+label_name+')', stacked=False)
        
    elif label_name == 'C':
        
        #plt.plot([i for i in range(-int(WIN-1), int(WIN+1))], fit_model, linewidth=2, color=Ccolorlist[0], label='GLMCC fitting('+label_name+')')
        plt.plot(x, fit_model, linewidth=1.5, color=Ccolorlist[0],label='GLMCC fitting('+label_name+')')
        plt.hist(c, bins=bin_num, range=(-1 * WIN, WIN), color=Ccolorlist[1], label='Cross correlogram('+label_name+')', stacked=False)
 
    plt.xlim(-50, 50)

def plt_W_GLMCC_with_A(c, par, tau, delay_synapse, pre_num, post_num, n, label_name, W_file, xlim=50):
    '''
    make histogram and GLMCC using matplotlib.
    A yellowgreen line express exp(a_k).
    A magenta line express excitatory exp(a_k + J_+ J_-).
    A cyan line express inhibitory exp(a_k + J_+ J_-).

    If you want to show graph, run "plt.show()" on Python.
    
    Input:
    Cross correlogram(list), parameter(list), tau(list), delay_synapse, pre cell number, post cell number, the number of cell, label_name(Python), W file

    ------------------------------------------------------------------------------------------
    ヒストグラムとGLMCCをmatplotlibを使って作る関数.
    黄緑色の線はexp(a_k)、マゼンタの線は興奮性に優位判定が出た時、シアンの線は抑制性に優位判定が出た時、それ以外のexp(a_k + J_+ + J_-)の線は灰色で表す。
    
    入力:
    Cross correlogram(list), パラメータ(list), tau(list), delay_synapse, 1番目のセルの番号, 2番目のセルの番号, セルデータの数, ラベル名(Python), Wファイル名
    '''

    if not(xlim == 50 or xlim == 49):
        print("error")

    bin_width = 1
    bin_num = int(2 * WIN / bin_width)

    fit_model = [math.exp(par[i]+par[NPAR-2]*func_f((i+1)*DELTA-WIN, delay_synapse, tau[0])+par[NPAR-1]*func_f(-(i+1)*DELTA+WIN, delay_synapse, tau[1])) for i in range(0, NPAR-2)]
    a_model = [math.exp(par[i]) for i in range(0, NPAR-2)]
    
    x = [i for i in range(-int(WIN-1), int(WIN+1))]
 
    colorlist = ['gray', 'yellowgreen', 'black']

    W = [[0 for i in range(n)] for j in range(n)]

    # read W file
    with open(W_file, 'r') as f:
        reader = csv.reader(f)

        i = 0
        for row in reader:
            for j in range(0, n):
                W[i][j] = float(row[j])
            i += 1

    f.close()
    
    pre = W[pre_num][post_num]
    post = W[post_num][pre_num]

    if pre > 0:
        pre_color = 'magenta'
    elif pre < 0:
        pre_color = 'cyan'
    else:
        pre_color = 'gray'

    if post > 0:
        post_color = 'magenta'
    elif post < 0:
        post_color = 'cyan'
    else:
        post_color = 'gray'

    if xlim == 50:

        plt.plot([i for i in range(-int(WIN-1), 1)], fit_model[0:50], linewidth=3.5, color=pre_color)
        plt.plot([i for i in range(0, int(WIN+1))], fit_model[49:], linewidth=3.5, color=post_color)

        #plt.plot(x, fit_model, linewidth=1, color=colorlist[0], label='GLMCC fitting('+label_name+')')
        plt.plot([i for i in range(-int(WIN-1), int(WIN+1))], a_model, linewidth=3.5, color=colorlist[1],alpha=1.0)

        plt.hist(c, bins=bin_num, range=(-1*WIN, WIN), color=colorlist[2], stacked=False)
         
        plt.xlim(-50, 50)
        #plt.xticks([-1, 0, 1])

    else:

        plt.plot([i for i in range(-int(WIN-1), 1)], fit_model[0:50], linewidth=3.5, color=pre_color)
        plt.plot([i for i in range(0, int(WIN+1))], fit_model[49:], linewidth=3.5, color=post_color)
        
        #plt.plot(x, fit_model, linewidth=1, color=Pycolorlist[0], label='GLMCC fitting('+label_name+')')
        plt.plot([i for i in range(-int(WIN-1), int(WIN+1))], a_model, linewidth=3.5,color=colorlist[1], alpha=1.0)

        hist_array = np.histogram(np.array(c), bins=bin_num, range=(-1*int(WIN), int(WIN)))
        hist_array = hist_array[0].tolist()

        for i in range(0, int(WIN-xlim)):
            del hist_array[int(WIN-i)]
            del hist_array[int(WIN-i-1)]
        
        plt.bar(range(-1*int(WIN)+int(WIN-xlim), int(WIN)-int(WIN-xlim)), hist_array, width=int(DELTA),color=colorlist[2])

        plt.xlim(-xlim, xlim)

    #plt.gca().set_aspect('equal', adjustable='box')

def plt_GLMCC_with_A(c, par, tau, delay_synapse, label_name):
    '''
    make histogram, GLMCC and GLMCC(only a_k) using matplotlib.
    

    If you want to show graph, run "plt.show()" on Python.
    
    Input:
    Cross correlogram(list), parameter(list), tau(list), delay_synapse, label_name(C or Python)

    --------------------------------------------------------------------------------------
    matplotlibを使って、ヒストグラム、GLMCC、GLMCC(a_kのみ)を描いた図を作る関数

    入力:
    Cross correlogram(list), パラメータ(list), tau(list), delay_synapse, label_name(C or Python)
    '''
    
    bin_width = 1
    bin_num = int(2 * WIN / bin_width)
    fit_model = [math.exp(par[i]+par[NPAR-2]*func_f((i+1)*DELTA-WIN, delay_synapse, tau[0])+par[NPAR-1]*func_f(-(i+1)*DELTA+WIN, delay_synapse, tau[1])) for i in range(0, NPAR-2)]
    a_model = [math.exp(par[i]) for i in range(0, NPAR-2)]
    x = [i for i in range(-int(WIN-2), int(WIN))]

    for i in range(1, -1, -1):
        del fit_model[int(WIN)-1+i]
        del a_model[int(WIN)-1+i]

    fit_model.insert(int(WIN-2)+delay_synapse, fit_model[int(WIN-2)+delay_synapse-1])
    fit_model.insert(int(WIN-2)-delay_synapse+1, fit_model[int(WIN-2)-delay_synapse+1])
    x.insert(int(WIN-2)+delay_synapse, delay_synapse)
    x.insert(int(WIN-2)-delay_synapse, -delay_synapse)

    #-1<t<1の範囲を残して結果を出すときこのコメントアウトを外す
    '''
    fit_model.insert(int(WIN-1)+delay_synapse, fit_model[int(WIN-1)+delay_synapse-1])
    fit_model.insert(int(WIN-1)-delay_synapse+1, fit_model[int(WIN-1)-delay_synapse+1])
    x = [i for i in range(-int(WIN-1), int(WIN+1))]
    x.insert(int(WIN-1)+delay_synapse, delay_synapse)
    x.insert(int(WIN-1)-delay_synapse+1, -delay_synapse)
    '''
    
    Pycolorlist = ['blue', 'green', 'cyan']
    Ccolorlist = ['red', 'orange', 'rosybrown']

    if label_name == 'Py':
        
        plt.plot(x, fit_model, linewidth=1, color=Pycolorlist[0], label='GLMCC fitting('+label_name+')')
        plt.plot([i for i in range(-int(WIN-2), int(WIN))], a_model, linewidth=1,linestyle='dashed', color=Pycolorlist[1], label='Fluctuation only('+label_name+')')

        hist_array = np.histogram(np.array(c), bins=bin_num, range=(-1*int(WIN), int(WIN)))
        hist_array = hist_array[0].tolist()
        del hist_array[int(WIN)]
        del hist_array[int(WIN-1)]
        
        plt.bar(range(-1*int(WIN)+1, int(WIN)-1), hist_array, width=int(DELTA),color=Pycolorlist[2])
        #plt.hist(c, bins=bin_num, range=(-1*WIN+1, WIN-1), color=Pycolorlist[2], label='Cross correlogram('+label_name+')', stacked=False)

    elif label_name == 'C':
        
        plt.plot(x, fit_model, linewidth=1.5, color=Ccolorlist[0], label='GLMCC fitting('+label_name+')')
        plt.plot([i for i in range(-int(WIN-2), int(WIN))], a_model, linewidth=1.5, linestyle='dashed', color=Ccolorlist[1], label='Fluctuation only('+label_name+')')

        hist_array = np.histogram(np.array(c), bins=bin_num, range=(-1*int(WIN), int(WIN)))
        hist_array = hist_array[0].tolist()
        del hist_array[int(WIN)]
        del hist_array[int(WIN-1)]

        plt.bar(range(-1*int(WIN)+1, int(WIN)-1), hist_array, width=int(DELTA), color=Ccolorlist[2])
        #plt.hist(c, bins=bin_num, range=(-1*WIN+1, WIN-1), color=Ccolorlist[2], label='Cross correlogram('+label_name+')', stacked=False)
        
    plt.xlim(-49, 49)

def plt_Ccode_GLMCC_with_A(filename1, filename2, tau, beta, T):
    '''
    make histogram, GLMCC and GLMCC(only a_k) using matplotlib. (C language)
    

    If you want to show graph, run "plt.show()" on Python.
    
    Input:
    cell file name1(string), cell file name2(string), tau(list), beta(float), T(string)
    ------------------------------------------------------------------------------------
    
    C言語でフィッティングした結果のGLMCCとa_kのみの結果とヒストグラムの図を出力する関数。

    入力:
    セルのファイル名1, セルのファイル名2, tau(list), beta(float), T(string)
    
    '''
    

    cmd = ['./Format', str(filename2), str(filename1), T]
    call_res = subprocess.check_call(cmd)
    #print(call_res)

    rel_spike_file = open("rel_spike.txt", "r")
    rel_spike = ''

    while True:
        line = rel_spike_file.readline()
        if not line:
            break
        rel_spike += line

    rel_spike_file.close()
    rel_spike = rel_spike.split()

    cmd = ['./Est_GLM', 'rel_spike.txt', str(tau[0]), str(tau[1]), str(beta), rel_spike[0], rel_spike[1]]
    call_res = subprocess.check_call(cmd)
    #print(call_res)
    
    par_file = open("J.txt", "r")
    par = ''

    while True:
        line = par_file.readline()
        if not line:
            break
        par += line

    par_file.close()
    
    par = par.split()

    if rel_spike[0] != par[0] and rel_spike[1] != par[1]:
        print("spike数があいません")
        return

    delay_synapse = int(float(par[6]))
    del rel_spike[0:2]
    par.append(par[2])
    par.append(par[3])
    del par[0:7]

    for i in range(0, len(rel_spike)):
        rel_spike[i] = float(rel_spike[i])

    for i in range(0, len(par)):
        par[i] = float(par[i])

    print('C par:')
    print(par)
    print(len(par))

    print('c n_sp: ' +str(len(rel_spike)))
    
    plt_GLMCC_with_A(rel_spike, par, tau, delay_synapse, 'C')

def plt_Ccode_GLMCC(filename1, filename2, tau, beta, T):
    '''
    make histogram and GLMCC using matplotlib.(C language)
    

    If you want to show graph, run "plt.show()" on Python.
    
    Input:
    cell file name1(string), cell file name2(string), tau(list), T(string)
    ------------------------------------------------------------------------
    C言語で計算したGLMCCの結果とヒストグラムの図を出力する関数.

    入力:
    セルのファイル名1, セルのファイル名2, tau(list), T(strin)

    '''

    cmd = ['./Format', str(filename2), str(filename1), T]
    call_res = subprocess.check_call(cmd)
    #print(call_res)

    rel_spike_file = open("rel_spike.txt", "r")
    rel_spike = ''

    while True:
        line = rel_spike_file.readline()
        if not line:
            break
        rel_spike += line

    rel_spike_file.close()
    rel_spike = rel_spike.split()

    cmd = ['./Est_GLM', 'rel_spike.txt', str(tau[0]), str(tau[1]), str(beta), rel_spike[0], rel_spike[1]]
    call_res = subprocess.check_call(cmd)
    #print(call_res)
    
    par_file = open("J.txt", "r")
    par = ''

    while True:
        line = par_file.readline()
        if not line:
            break
        par += line

    par_file.close()
    
    par = par.split()

    if rel_spike[0] != par[0] and rel_spike[1] != par[1]:
        print("spike数があいません")
        return

    delay_synapse = int(float(par[6]))
    del rel_spike[0:2]
    par.append(par[2])
    par.append(par[3])
    del par[0:7]

    for i in range(0, len(rel_spike)):
        rel_spike[i] = float(rel_spike[i])

    for i in range(0, len(par)):
        par[i] = float(par[i])

    print('C par:')
    print(par)
    print(len(par))
    
    plt_GLMCC(rel_spike, par, tau, delay_synapse, 'C')

def print_usage():
    print("Cross correlogramを描く")
    print("Usage: python3 glm.py CC f_pre.txt f_post.txt T(s)")
    print()
    print("GLMのフィッティングの結果をCross correlogramと共に描く")
    print("Usage: python3 glm.py GLM f_pre.txt f_post.txt tau_+ tau_- gamma T(s)")
    print()
    print("C言語のフィッティングとPyのフィッティングを両方描く")
    print("Usage: python3 glm.py GLM-Py-C f_pre.txt f_post.txt tau_+ tau_- gamma T(s)")
    print()
    print("GLM-Py-Cの結果にexp(a_k)の結果を描く")
    print("Usage: python3 glm.py GLM-with-A f_pre.txt f_post.txt tau_+ tau_- gamma T(s)")
    print()
    print("Wのグラフをだす")
    print("Usage: python3 glm.py W_plot W_file.csv f_plot.txt cell_num T (no or cell_dir)")
    print()
    print("GLMのフィッティングの結果をCross correlogramと共に描く(優位判定を行う)")
    print("Usage: python3 glm.py W-GLM-with-A f_pre.txt f_post.txt tau_+ tau_- gamma T(s) cell_num W_file.csv xlim")
    print("---------------------------------------------------------------------------")
    var_exp = '''
    変数の説明:
    f_pre.txt: 最初のセルファイル名
    f_post.txt: 二番目のセルファイル名
    W_file.csv: PSPの行列のファイル名
    f_post.txt: Wのプロット図を出力するために一時的にデータを保存するファイル
    tau_+, tau_-: tauの値
    gamma: gamma
    T(s): timescale
    cell_num: セルデータのファイルの総数
    xlim: ヒストグラムの時間軸の範囲
          もしWIN(50)を入れればそのまま出力
          xlimを入れれば-(WIN-xlim)<t<(WIN-xlim)の範囲のヒストグラムのデータが消える
          (つまり全体の範囲が-xlim<t<xlimに変化する)
    '''

    print(var_exp)

if __name__ == '__main__':

    args = sys.argv

    if len(args) == 5:
        if args[1] == 'CC':
            # python3 glm.py CC f_pre.txt f_post.txt T
            T = float(args[4])
            cc_list = linear_crossCorrelogram(args[3], args[2], T)
            bin_width = DELTA
            bin_num = int(2 * WIN / bin_width)
            plt.hist(cc_list[0], bins=bin_num, range=(-1 * WIN,WIN), stacked=False)
            plt.title(args[2][:-4]+"_"+args[3][:-4]+" Histgram")
             
            plt.xlim(-50, 50)
            plt.legend(fontsize=10)
            plt.savefig("GLMCC_"+args[2][:-4]+"_"+args[3][:-4]+".png")
            plt.show()
            plt.close()

        else:
            print_usage()

    elif len(args) == 7:
        if args[1] == 'W_plot':
            if args[6] == 'no':              
                plot_3d(args[3], args[2], int(float(args[4])))
                plot_PSP_est(args[2], int(float(args[5])), int(float(args[4])))
                
            else:
                divide_into_E_I(args[3], int(float(args[4])), args[6])
                plot_3d("sorted_W.csv", args[2], int(float(args[4])))
                plot_PSP_est(args[2], int(float(args[5])), int(float(args[4])))
                subprocess.call( [ 'rm', 'sorted_W.csv'])
                
        else:
            print_usage()

    elif len(args) == 8:
        if args[1] == 'GLM' or args[1] == 'LR':
            LR = False
            if args[1] == 'LR':
                LR = True
            T = float(args[7])
            cc_list = linear_crossCorrelogram(args[3], args[2], T)
            tau = [0, 0]
            tau[0] = float(args[4])
            tau[1] = float(args[5])
            gamma = float(args[6])
            beta = 2.0/gamma
            delay_synapse = 1
            par, log_pos, log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
            for i in range(2, 5):
                tmp_par, tmp_log_pos, tmp_log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], i)
                if (not LR and tmp_log_pos > log_pos) and (LR and tmp_log_likelihood > log_likelihood):
                    log_pos = tmp_log_pos
                    log_likelihood = tmp_log_likelihood
                    par = tmp_par
                    delay_synapse = i

            print('delay synapse:')
            print(delay_synapse)
            plt_GLMCC(cc_list[0], par, tau, delay_synapse, 'Py')
            plt.legend(fontsize=10)
            plt.title(args[2][:-4]+"_"+args[3][:-4]+" Histgram")
            plt.savefig("GLMCC_"+args[2][:-4]+"_"+args[3][:-4]+".png")
            plt.show()
            plt.close()

        elif args[1] == 'GLM-Py-C':
            T = int(float(args[7]))
            cc_list = linear_crossCorrelogram(args[3], args[2], T)
            tau = [0, 0]
            tau[0] = float(args[4])
            tau[1] = float(args[5])
            gamma = float(args[6])
            beta = 2.0/gamma
            delay_synapse = 1
            par, log_pos = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
            for i in range(2, 5):
                tmp_par, tmp_log_pos = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], i)
                if tmp_log_pos > log_pos:
                    log_pos = tmp_log_pos
                    par = tmp_par
                    delay_synapse = i
                    

            #Connection parameters
            nb = int(WIN/DELTA)
            cc_0 = [0 for i in range(2)]
            max = [0 for i in range(2)]
            for i in range(2):
                cc_0[i] = 0
                max[i] = int(tau[i] + 0.1)

                if i == 0:
                    for k in range(max[i]):
                        cc_0[i] += np.exp(par[nb+int(delay_synapse)+k])
                if i == 1:
                    for k in range(max[i]):
                        cc_0[i] += np.exp(par[nb-int(delay_synapse)-k])
                cc_0[i] = cc_0[i]/max[i]
                n12 = tau[i]*cc_0[i]
                if n12 <= 10:
                    par[NPAR-2+i] = 0
                
            print('python par:')
            print(par)
            print('delay synapse')
            print(delay_synapse)
            plt_Ccode_GLMCC(args[3], args[2], tau, beta, args[7])
            plt_GLMCC(cc_list[0], par, tau, delay_synapse, 'Py')
            plt.legend(fontsize=10)
            plt.title(args[2][:-4]+"_"+args[3][:-4]+" Histgram(Py and C)")
            plt.savefig("GLMCC_Py-C_"+args[2][:-4]+"_"+args[3][:-4]+".png")
            plt.show()
            plt.close()

        elif args[1] == 'GLM-with-A':
            T = int(float(args[7]))
            cc_list = linear_crossCorrelogram(args[3], args[2], T)
            tau = [0, 0]
            tau[0] = float(args[4])
            tau[1] = float(args[5])
            gamma = float(args[6])
            beta = 2.0/gamma
            delay_synapse = 1
            par, log_pos = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
            print(log_pos)
            print(par)
            for i in range(2, 5):
                tmp_par, tmp_log_pos = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], i)
                print(tmp_log_pos)
                print(tmp_par)
                if tmp_log_pos > log_pos:
                    log_pos = tmp_log_pos
                    par = tmp_par
                    delay_synapse = i
            
            #Connection parameters
            nb = int(WIN/DELTA)
            cc_0 = [0 for i in range(2)]
            max = [0 for i in range(2)]
            Jmin = [0 for i in range(2)]
            for i in range(2):
                cc_0[i] = 0
                max[i] = int(tau[i] + 0.1)

                if i == 0:
                    for k in range(max[i]):
                        cc_0[i] += np.exp(par[nb+int(delay_synapse)+k])
                if i == 1:
                    for k in range(max[i]):
                        cc_0[i] += np.exp(par[nb-int(delay_synapse)-k])
                cc_0[i] = cc_0[i]/max[i]

                Jmin[i] = math.sqrt(16.3/ tau[i]/ cc_0[i])
                n12 = tau[i]*cc_0[i]
                if n12 <= 10:
                    par[NPAR-2+i] = 0

            print('python par:')
            print(par)
            print('delay synapse')
            print(delay_synapse)
            plt_Ccode_GLMCC_with_A(args[3], args[2], tau, beta, args[7])
            plt_GLMCC_with_A(cc_list[0], par, tau, delay_synapse, 'Py')
            plt.legend(fontsize=10)
            plt.title(args[2][:-4]+"_"+args[3][:-4]+" Histgram(Py and C)")
            plt.savefig("GLMCC_Py-C_"+args[2][:-4]+"_"+args[3][:-4]+"(a_k).png")
            plt.show()
            plt.close()

        else:
            print_usage()

    elif len(args) == 11:
        if args[1] == 'W-GLM-with-A':
            T = int(float(args[7]))
            cc_list = linear_crossCorrelogram(args[3], args[2], T)
            tau = [0, 0]
            tau[0] = float(args[4])
            tau[1] = float(args[5])
            gamma = float(args[6])
            beta = 2.0/gamma
            delay_synapse = 1
            par, log_pos = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
            print(log_pos)
            print(par)
            for i in range(2, 5):
                tmp_par, tmp_log_pos = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], i)
                print(tmp_log_pos)
                print(tmp_par)
                if tmp_log_pos > log_pos:
                    log_pos = tmp_log_pos
                    par = tmp_par
                    delay_synapse = i
            
            #Connection parameters
            nb = int(WIN/DELTA)
            cc_0 = [0 for i in range(2)]
            max = [0 for i in range(2)]
            Jmin = [0 for i in range(2)]
            for i in range(2):
                cc_0[i] = 0
                max[i] = int(tau[i] + 0.1)

                if i == 0:
                    for k in range(max[i]):
                        cc_0[i] += np.exp(par[nb+int(delay_synapse)+k])
                if i == 1:
                    for k in range(max[i]):
                        cc_0[i] += np.exp(par[nb-int(delay_synapse)-k])
                cc_0[i] = cc_0[i]/max[i]

                Jmin[i] = math.sqrt(16.3/ tau[i]/ cc_0[i])
                n12 = tau[i]*cc_0[i]
                if n12 <= 10:
                    par[NPAR-2+i] = 0

            print('python par:')
            print(par)
            print('delay synapse')
            print(delay_synapse)
            plt.figure(figsize=(5,5))
            if int(args[10]) == 50:
                plt_W_GLMCC_with_A(cc_list[0], par, tau, delay_synapse, int(args[3][4:-4]), int(args[2][4:-4]), int(args[8]), 'Py', args[9])
            else:
                plt_W_GLMCC_with_A(cc_list[0], par, tau, delay_synapse, int(args[3][4:-4]), int(args[2][4:-4]), int(args[8]), 'Py', args[9], int(args[10]))
            plt.legend(fontsize=10)
            plt.xticks(np.arange(-int(args[10]), int(args[10]) + 1, 50))
            #plt.figure(figsize=(10,10))
            plt.title(args[2][:-4]+"_"+args[3][:-4]+" Histgram")
            plt.savefig("GLMCC_"+args[2][:-4]+"_"+args[3][:-4]+"_wid_"+args[10]+".png")
            plt.show()
            plt.close() 

    else:
        print_usage()
