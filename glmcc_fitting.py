# coding: utf-8
from glmcc import *
import matplotlib.pyplot as plt
import subprocess as proc
import sys

args = sys.argv

if len(args) != 7:
    print("python glmcc_fitting.py cell_num cell_dir (sim or exp) Wfile_pass (all or sgn) (GLM or LR)")
    exit(0)

cell_num = int(args[1])
LR = False
if args[6] == "LR":
    LR = True

plt.figure(figsize=(50,50),dpi=600)
for i in range(0, cell_num):
    for j in range(0, cell_num):
        
        if i == j:
            continue
        
        filename1 = args[2]+'/cell'+str(i)+'.txt'
        filename2 = args[2]+'/cell'+str(j)+'.txt'
        print(filename1+' '+filename2)
        T = 5400

        #plot cross_correlogram
        cc_list = linear_crossCorrelogram(filename1, filename2, T)

        #set tau
        tau = [4, 4]
        beta = 4000

        #Fitting a GLM
        if args[3] == 'exp':
            delay_synapse = 1
            par, log_pos, log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
            #print('J+: '+str(par[NPAR-2])+' J-: '+str(par[NPAR-1]))
        
            for m in range(2, 5):
                tmp_par, tmp_log_pos, tmp_log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], m)
                if (not LR and tmp_log_pos > log_pos) or (LR and tmp_log_likelihood > log_likelihood):
                    log_pos = tmp_log_pos
                    log_likelihood = tmp_log_likelihood
                    par = tmp_par
                    delay_synapse = m

        elif args[3] == 'sim':
            delay_synapse = 3
            par, log_pos, log_likelihood = GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
            #print('J+: '+str(par[NPAR-2])+' J-: '+str(par[NPAR-1]))

        """
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

        # We examine whether the pair is connected significantly for plotting only the signifinicant pairs.
        if args[5] == 'sgn':
            # read W file
            W = [[0 for l in range(cell_num)] for m in range(cell_num)]
            with open(args[4], 'r') as f:
                reader = csv.reader(f)

                l = 0
                for row in reader:
                    for m in range(0, cell_num):
                        W[l][m] = float(row[m])
                    l += 1

            f.close()

            # We do not plot for the nonsignificant pairs
            if W[j][i] == 0:
                continue
        """

        # Plot the cross-correlerogram. 実験データとシミュレーションデータの処理は-1<t<1を取り除くため、分けて処理する. 
        plt.subplot(cell_num, cell_num, i*cell_num+j+1)

        #グラフのサイズ変更
        #plt.figure(figsize=(2,2),dpi=40)
        
        if args[3] == 'sim':
            plt_W_GLMCC_with_A(cc_list[0], par, tau, delay_synapse, i, j, cell_num, 'Py', args[4])
        elif args[3] == 'exp':
            plt_W_GLMCC_with_A(cc_list[0], par, tau, delay_synapse, i, j, cell_num, 'Py', args[4], 49)

        #グラフを正方形にする
        #plt.subplot().set_aspect('equal')
        
        plt.legend(fontsize=10)
        #plt.show()
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        

#plt.figure(figsize=(50, 50))
plt.savefig("allCC.png")
plt.close()
