#Author: Junichi haruna
'''
Estimate PSPs from Data.

Input : Data file and the number of data

Command : python3 Est_Data.py (Data file name) (the number of data) (sim or exp(simulation or experiment))

example command : python3 Est_Data.py simulation_data 20 sim

'''

from glmcc import *
import sys
import subprocess as proc

args = sys.argv

if len(args) != 5:
    print("Usage: python3 Est_Data.py (Data file name) (the number of data) (sim or exp) (GLM or LR)")
    exit(0)

DataFileName = args[1]
DataNum = int(args[2])
mode = args[3]
LR = False
beta = 4000
if args[4] == "LR":
    LR = True
    beta = 10000

for i in range(0, DataNum):
    for j in range(0, i):
        filename1 = DataFileName+'/cell'+str(i)+'.txt'
        filename2 = DataFileName+'/cell'+str(j)+'.txt'
        print(filename1+' '+filename2)
        T = 5400

        #Make cross_correlogram
        cc_list = linear_crossCorrelogram(filename1, filename2, T)

        #set tau
        tau = [4, 4]

        #Fitting a GLM
        if mode == 'sim':
            delay_synapse = 3
            par, log_pos, log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
        elif mode == 'exp':
            log_pos = 0
            log_likelihood = 0
            for m in range(1, 5):
                tmp_par, tmp_log_pos, tmp_log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], m)
                if m == 1 or (not LR and tmp_log_pos > log_pos) or (LR and tmp_log_likelihood > log_likelihood):
                    log_pos = tmp_log_pos
                    log_likelihood = tmp_log_likelihood
                    par = tmp_par
                    delay_synapse = m
        else:
            print("Input error: You must write sim or exp in mode")
            print("Usage: python3 Est_Data.py (Data file name) (the number of data) (sim or exp)")
            exit(0)
                    

        #Connection parameters
        nb = int(WIN/DELTA)
        cc_0 = [0 for l in range(2)]
        max = [0 for l in range(2)]
        Jmin = [0 for l in range(2)]
        for l in range(2):
            cc_0[l] = 0
            max[l] = int(tau[l] + 0.1)
            
            if l == 0:
                for m in range(max[l]):
                    cc_0[l] += np.exp(par[nb+int(delay_synapse)+m])
            if l == 1:
                for m in range(max[l]):
                    cc_0[l] += np.exp(par[nb-int(delay_synapse)-m])

            cc_0[l] = cc_0[l]/max[l]
                    
            Jmin[l] = math.sqrt(16.3/ tau[l]/ cc_0[l])
            n12 = tau[l]*cc_0[l]
            if n12 <= 10:
                par[NPAR-2+l] = 0
        D1 = 0
        D2 = 0
        if LR:
            tmp_par, tmp_log_pos, log_likelihood_p = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond = 1)
            tmp_par, tmp_log_pos, log_likelihood_n = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond = 2)
            D1 = log_likelihood - log_likelihood_p
            D2 = log_likelihood - log_likelihood_n

        #Output J
        J_f = open("J_py_"+str(T)+".txt", 'a')
        J_f.write(str(i)+' '+str(j)+' '
                  +str(round(par[NPAR-1], 6))+' '+str(round(par[NPAR-2], 6))+' '
                  +str(round(Jmin[1], 6))+' '+str(round(Jmin[0], 6))+' '
                  +str(round(D2, 6))+' '+str(round(D1, 6))+'\n')
        J_f.close()

n = DataNum
scale = 1.277
z_a = 15.14

#Read the required J file and create the resul file
J_f = open("J_py_"+str(T)+".txt", 'r')
J_f_list = J_f.readlines()
W_f = open("W_py_"+str(T)+".csv", 'w')
W = [[0 for i in range(n)] for j in range(n)]

#calculate W
for i in range(0, len(J_f_list)):
    J_f_list[i] = J_f_list[i].split()
    
    J_f_list[i][0] = int(float(J_f_list[i][0])) #pre
    J_f_list[i][1] = int(float(J_f_list[i][1])) #post
    J_f_list[i][2] = float(J_f_list[i][2])      #J_+
    J_f_list[i][3] = float(J_f_list[i][3])      #J_-
    J_f_list[i][4] = float(J_f_list[i][4])      #J_min_+
    J_f_list[i][5] = float(J_f_list[i][5])      #J_min_-
    J_f_list[i][6] = float(J_f_list[i][6])      #D_+
    J_f_list[i][7] = float(J_f_list[i][7])      #D_-
    
    if not LR:
        W[J_f_list[i][0]][J_f_list[i][1]] = round(calc_PSP(J_f_list[i][2], J_f_list[i][4]*scale), 6)
        W[J_f_list[i][1]][J_f_list[i][0]] = round(calc_PSP(J_f_list[i][3], J_f_list[i][5]*scale), 6)
    else:
        W[J_f_list[i][0]][J_f_list[i][1]] = round(calc_PSP_LR(J_f_list[i][2], J_f_list[i][6], z_a), 6)
        W[J_f_list[i][1]][J_f_list[i][0]] = round(calc_PSP_LR(J_f_list[i][3], J_f_list[i][7], z_a), 6)

#write W
for i in range(0, n):
    for j in range(0, n):
        W_f.write(str(W[i][j]))   # v1909,   JH
        if j == n-1:
            W_f.write('\n')
        else:
            W_f.write(', ')

#remove J file

# debug
cmd = ['rm', "J_py_"+str(T)+".txt"]
proc.check_call(cmd)

