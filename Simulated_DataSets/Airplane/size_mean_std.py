import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def compute_size():
    mean_list = []
    std_list = []
    step = 5
    for size_pro in size_range:
        if size_pro < 0:
            low = rc_low + size_pro*step
            high = rc_high + size_pro*step
        else:
            low = rc_low + size_pro*step
            high = rc_high + size_pro*step
        size_unif = np.random.uniform(low, high, num_aft)
        mean_size = np.mean(size_unif)
        std_size = np.std(size_unif)
        mean_list.append(mean_size)
        std_list.append(std_size)
    mean_list = np.around(mean_list, decimals=3)
    std_list = np.around(std_list, decimals=3)
    print('mean_list ', mean_list)
    print('std list', std_list)
    df_rgb = pd.DataFrame(columns=['Version', 'size_mean', 'size_std'], index=None)

    for ix in range(8):
        vix = ix + base_ix
        df_rgb = df_rgb.append({'Version':vix, 'size_mean':mean_list[ix], 'size_std':std_list[ix]}, ignore_index=True)
    print(df_rgb)
    if rare_cls == 1:
        mode = 'w'
    else:
        mode = 'a'
    with pd.ExcelWriter(os.path.join('result_output', 'rare_class_size_bias.xlsx'), mode=mode) as writer:
        df_rgb.to_excel(writer, sheet_name='rc{}'.format(rare_cls), index=False)


def generate_normal(sd, mu, sigma, size):
    np.random.seed(sd)
    u_list = np.random.random(size=size)
    v_list = np.random.random(size=size)
    gs = []
    gsstr = ""
    for ix in range(size):
        u = u_list[ix]
        v = v_list[ix]
        # z1 = np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v)
        z2 = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
        print('u', u, 'v', v, 'z2', z2)
    # x1 = mu + z1 * sigma
        x2 = mu + z2 * sigma
        gs.append(x2)
        gsstr += '{:.2f};'.format(x2)

    return gs, gsstr

def gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft):
    np.random.seed(1)
    mu = np.array(mu)
    save_dir = 'result_output/RC_size/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # df_size = pd.DataFrame(columns=['Version', 'mean_body', 'mean_wing', 'size_sqsigma'], index=None)
    f_txt = open(os.path.join(save_dir, 'RC{}_size.txt'.format(rare_cls)), 'w')
    gs = []
    for ix, ssig in enumerate(ssig_list):
        # diag =  ssig * np.diag(mu)
        diag =  np.diag(np.power(ssig * mu, 2))
        gs = np.random.multivariate_normal(mu, diag, num_aft)
        # plt.hist(gs, 30)
        # a = np.random.choice(gs[:, 0], 450*2)
        # print("mean, std", np.mean(a), np.std(a))
        # plt.hist(a, 30)
        # plt.show()
        body_str = ''
        wing_str = ''
        for jx in range(gs.shape[0]):
            body_str += '{:.2f};'.format(gs[jx, 0])
            wing_str += '{:.2f};'.format(gs[jx, 1])

        f_txt.write('@Hidden\nattr body{}= "{}"\n\n'.format(int(ssig*100), body_str))
        f_txt.write('@Hidden\nattr wing{}= "{}"\n\n'.format(int(ssig*100), wing_str))
        # df_size = df_size.append({'Version':ix, 'mean_body':mu[0], 'mean_wing':mu[1], 'size_sqsigma':ssig}, ignore_index=True)
    f_txt.close()


def gaussian_disribution_of_size_mu(mu_ori, ssig, mu_times, rare_cls, num_aft):
    np.random.seed(1)
    mu_ori = np.array(mu_ori)
    save_dir = 'result_output/RC_size/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # df_size = pd.DataFrame(columns=['Version', 'mean_body', 'mean_wing', 'size_sqsigma'], index=None)
    f_txt = open(os.path.join(save_dir, 'v2-RC{}_size_mu.txt'.format(rare_cls)), 'w')
    gs = []
    for ix, ts in enumerate(mu_times):
        # diag =  ssig * np.diag(mu)
        mu = mu_ori*(1+ts)
        print('mu', mu)
        diag =  np.diag(np.power(ssig * mu, 2))
        # gs = np.random.multivariate_normal(mu, diag, num_aft) # before 5/10/2021
        gs = np.random.multivariate_normal(mu, diag, num_aft + 3000)
        body_str = ''
        wing_str = ''
        cnt = 0
        for jx in range(gs.shape[0]):
            if np.any(gs[jx] <= 0):
                continue
            body_str += '{:.2f};'.format(gs[jx, 0])
            wing_str += '{:.2f};'.format(gs[jx, 1])
            cnt += 1
            if cnt == num_aft:
                break

        f_txt.write('@Hidden\nattr body_mu{}= "{}"\n\n'.format(int(ts*100), body_str))
        f_txt.write('@Hidden\nattr wing_mu{}= "{}"\n\n'.format(int(ts*100), wing_str))
    f_txt.close()



def gaussian_disribution_of_size_mu_CC(mu_ori, ssig, mu_times, common_cls, num_aft):
    np.random.seed(1)
    mu_ori = np.array(mu_ori)
    save_dir = 'result_output/CC_size/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # df_size = pd.DataFrame(columns=['Version', 'mean_body', 'mean_wing', 'size_sqsigma'], index=None)
    f_txt = open(os.path.join(save_dir, 'CC{}_size_mu.txt'.format(common_cls)), 'w')
    gs = []
    for ix, ts in enumerate(mu_times):
        # diag =  ssig * np.diag(mu)
        mu = mu_ori*(1+ts)
        print('mu', mu)
        diag =  np.diag(np.power(ssig * mu, 2))
        gs = np.random.multivariate_normal(mu, diag, num_aft)
        body_str = ''
        wing_str = ''
        for jx in range(gs.shape[0]):
            body_str += '{:.2f};'.format(gs[jx, 0])
            wing_str += '{:.2f};'.format(gs[jx, 1])

        f_txt.write('@Hidden\nattr body_mu{}= "{}"\n\n'.format(int(ts*100), body_str))
        f_txt.write('@Hidden\nattr wing_mu{}= "{}"\n\n'.format(int(ts*100), wing_str))
    f_txt.close()
    plot_sigma_size_dist(save_dir, size_file_name, mu_ori)

def gaussian_disribution_of_body_wing_CC(ccid, num_aft, size_file, best_step=0):
    df_size = pd.read_csv(size_file, sep='\t')
    body = df_size['Body'].to_numpy()
    wing = df_size['Wing'].to_numpy()
    mu_body = np.around(np.mean(body), decimals=1)
    mu_wing = np.around(np.mean(wing), decimals=1)
    std_body = np.around(np.std(body), decimals=2)
    std_wing = np.around(np.std(wing), decimals=2)
    print('mu_body, std_body, std/mu', mu_body, std_body, std_body/mu_body/4)
    print('mu_wing, std_wing, std/mu', mu_wing, std_wing, std_wing/mu_wing/4)
    step_body = np.around(std_body/mu_body/4, decimals=2)
    step_wing = np.around(std_wing/mu_wing/4, decimals=2)
    body_ssig_list = [i*step_body for i in range(5)]
    wing_ssig_list = [i*step_wing for i in range(5)]
    if ccid==1:
        body_ssig_list = [step_body*best_step]
        wing_ssig_list = [step_wing*best_step]
    else:
        body_ssig_list = [step_body*best_step]
        wing_ssig_list = [step_wing*best_step]

    print('body_list', body_ssig_list)
    print('wing_list', wing_ssig_list)

    np.random.seed(1)
    save_dir = 'Simulated_DataSets/Airplane/CC_size/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_txt = open(os.path.join(save_dir, 'syn_CC{}_size.txt'.format(ccid)), 'w')
    mu = np.array([mu_body, mu_wing])
    for ix in range(len(body_ssig_list)):
        ssig = [body_ssig_list[ix]*mu_body, wing_ssig_list[ix]*mu_wing]
        diag =  np.diag(np.power(ssig, 2))
        gs = np.random.multivariate_normal(mu, diag, num_aft)
        # plt.hist(gs, 30)
        # a = np.random.choice(gs[:, 0], 450*2)
        # print("mean, std", np.mean(a), np.std(a))
        # plt.hist(a, 30)
        # plt.show()
        body_str = ''
        wing_str = ''
        for jx in range(gs.shape[0]):
            body_str += '{:.2f};'.format(gs[jx, 0])
            wing_str += '{:.2f};'.format(gs[jx, 1])

        f_txt.write('attr body{}= "{}"\n\n'.format(int(body_ssig_list[ix]*100), body_str))
        f_txt.write('attr wing{}= "{}"\n\n'.format(int(wing_ssig_list[ix]*100), wing_str))
    f_txt.close()


def increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu_ori, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='RC'):
    np.random.seed(1)
    mu_ori = np.array(mu_ori)
    save_dir = 'result_output/{}_increased_size_sigma_based_on_optimal_ssig/'.format(cmt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # df_size = pd.DataFrame(columns=['Version', 'mean_body', 'mean_wing', 'size_sqsigma'], index=None)
    size_file_name = 'v6-{}{}_size_mu.txt'.format(cmt, rare_cls)
    f_txt = open(os.path.join(save_dir, size_file_name), 'w')
    gs = []
    
    mu = mu_ori*(1+mu_time)
    print('mu', mu)
    delta_mu = mu_ori*mu_time
    for ss in sig_size_list:
        # if not np.any(ssig):
        #     # diag =  np.diag(np.power(ss * mu, 2)) # v2
        #     diag =  np.diag(np.power([ss*2, ss*2], 2)) # v3 from 4/26/2021
        # else:
        #     diag =  np.diag(np.power(ssig * (1+ss) * mu, 2))
        
        std = delta_mu * ss
        print('std', std)
        diag =  np.diag(np.power(std, 2)) # v4 from 5/2/2021
        print('diag', diag)
        gs = np.random.multivariate_normal(mu, diag, num_aft + 3000)
        body_str = ''
        wing_str = ''
        jc = 0
        for jx in range(gs.shape[0]):
            if np.all(gs[jx]>0):
                body_str += '{:.2f};'.format(gs[jx, 0])
                wing_str += '{:.2f};'.format(gs[jx, 1])
                jc += 1
            if jc == num_aft:
                break
        print('jc', jc)
        # lbl = int(ss) # v3 int(ss*100)
        lbl =  int(ss*100)
        f_txt.write('@Hidden\nattr body_mu{}= "{}"\n\n'.format(lbl, body_str)) 
        f_txt.write('@Hidden\nattr wing_mu{}= "{}"\n\n'.format(lbl, wing_str)) 
        
    f_txt.close()
    plot_sigma_size_dist(save_dir, size_file_name, mu_ori)


    
def plot_sigma_size_dist(sigma_path, sigma_file, mu):
    df_rc = pd.read_csv(os.path.join(sigma_path, sigma_file), header=None)
    plt.figure()
    fig, axs = plt.subplots(1,2, sharey=True)
    color_list = ['c', 'r', 'g', 'b']
    for i in range(1, 4):
        b = df_rc.loc[1+4*i, 0].split('=')[1]
        b = b.strip(' " ').split(';')[:-1]
        w = df_rc.loc[3+4*i, 0].split('=')[1]
        w = w.strip(' " ').split(';')[:-1]
        # print(r[-2:])
        b = [float(b) for b in b]
        w = [float(w) for w in w]
        
        # print('w', len(w))

        bs_list = []
        ws_list = []
        for j in range(len(b)):
            # bs = np.abs(b[j]-mu[0])
            # ws = np.abs(w[j]-mu[1])
            bs = (b[j]-mu[0])
            ws = (w[j]-mu[1])
            bs_list.append(bs)
            ws_list.append(ws)
        
        # lgd = f"$\Delta \sigma_s=$ {int(sig_size_list[i]*100)}"
        # lgd = f"$j=$ {int(sig_size_list[i])}"
        lgd = "$j=$ {:.2f}".format(sig_size_list[i])
        axs[0].hist(bs_list, bins=50, color=color_list[i], alpha=0.4, label= lgd )
        axs[0].set_xlabel('body size distance to $\mu_s$')
        axs[1].hist(ws_list, bins=50, color=color_list[i], alpha=0.4, label= lgd)
        axs[1].set_xlabel('wing size distance to $\mu_s$')
    plt.legend()
    title = sigma_file.split('_size')[0]
    fig.suptitle(title)
    plt.savefig(os.path.join(sigma_path, 'sigma_' + sigma_file.replace('.txt', '.png')))



if __name__ == '__main__':
    
    ################ Delta simga size, sigma color
    # sig_size_list = np.array([0, 0.1, 0.2, 0.3], dtype=np.float32)*2
    # sig_size_list = np.array([0, 0.2, 0.4, 0.6], dtype=np.float32)*3
    # sig_size_list = np.array([0, 0.1, 0.2, 0.3], dtype=np.float32)*5
    # sig_size_list = np.array([0, 1, 2, 3], dtype=np.float32) # v4
    # sig_size_list = np.array([0, 0.33, 0.66, 1], dtype=np.float32) # v6
    # num_aft = int(450*7*3)
    
    ## rc1
    # num_aft = int(450*7*3)
    # common_cls = 1
    # mu = [13, 7]
    # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)

    # num_aft = int(450*7*3)
    # rare_cls = 1
    # mu = [13, 7]
    # mu_times  = [0.1, 0.2, 0.3]
    # ssig = 0.03
    # gaussian_disribution_of_size_mu(mu, ssig, mu_times, rare_cls, num_aft)
 
    
    ####### increase sigma size based on large mu_size=mu*(1+0.3)
    # num_aft = int(450*7*3)
    # rare_cls = 1
    # print(f'-------------{rare_cls}------------')
    # mu = [13, 7]
    # mu_time  = 0.3
    # ssig = 0.03
    # increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='RC')

    # # # # ## rc2
    # # # num_aft = int(450*7*3)
    # # # rare_cls = 2
    # # # mu = [38.3, 33]
    # # # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # # # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)

    # # num_aft = int(450*7*3)
    # # rare_cls = 2
    # # mu = [38.3, 33]
    # # mu_times  = [0.1, 0.2, 0.3]
    # # ssig = 0.03
    # # gaussian_disribution_of_size_mu(mu, ssig, mu_times, rare_cls, num_aft)
    
    # ####### increase sigma size based on large mu_size=mu*(1+0.3)
    # # num_aft = int(450*7*3)
    # rare_cls =2
    # print(f'-------------{rare_cls}------------')
    # mu = [38.3, 33]
    # mu_time  = 0.3
    # ssig = 0.03
    # increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='RC')

    # # #
    # # # ## rc3
    # # # num_aft = int(450*7*3)
    # # # rare_cls = 3
    # # # mu = [25.7, 25.5]
    # # # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # # # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)

    # # num_aft = int(450*7*3)
    # # rare_cls = 3
    # # mu = [25.7, 25.5]
    # # mu_times  = [0.1, 0.2, 0.3]
    # # ssig = 0.09
    # # gaussian_disribution_of_size_mu(mu, ssig, mu_times, rare_cls, num_aft)

    # ####### increase sigma size based on large mu_size=mu*(1+0.3)
    # # num_aft = int(450*7*3)
    # rare_cls =3
    # print(f'-------------{rare_cls}------------')
    # mu = [25.7, 25.5]
    # mu_time  = 0.3
    # ssig = 0.09
    # increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='RC')


    # # #
    # # # ## rc4
    # # # num_aft = int(450*7*3)
    # # # rare_cls = 4
    # # # mu = [31, 39]
    # # # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # # # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)

    # num_aft = int(450*7*3)
    # rare_cls = 4
    # mu = [31, 39]
    # mu_times  = [0.1, 0.2, 0.3]
    # # ssig = 0.12 # before 5/10/2021
    # ssig = 0.03
    # gaussian_disribution_of_size_mu(mu, ssig, mu_times, rare_cls, num_aft)
 
    # ####### increase sigma size based on large mu_size=mu*(1+0.3)
    # # num_aft = int(450*7*3)
    # rare_cls = 4
    # print(f'-------------{rare_cls}------------')
    # mu = [31, 39]
    # mu_time  = 0.3
    # # ssig = 0.12 # before 5/10/2021
    # ssig = 0.03
    # increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='RC')


    # # ## rc5
    # # # num_aft = int(450*7*3)
    # # # rare_cls = 5
    # # # mu = [8.3, 19.9]
    # # # ssig_list = [0, 0.03, 0.06, 0.09, 0.12]
    # # # gaussian_disribution_of_size(mu, ssig_list, rare_cls, num_aft)

    # # num_aft = int(450*7*3)
    # # rare_cls = 5
    # # mu = [8.3, 19.9]
    # # mu_times  = [0.1, 0.2, 0.3]
    # # ssig = 0.03
    # # gaussian_disribution_of_size_mu(mu, ssig, mu_times, rare_cls, num_aft)

    # ####### increase sigma size based on large mu_size=mu*(1+0.3)
    # # num_aft = int(450*7*3)
    # rare_cls = 5
    # print(f'-------------{rare_cls}------------')
    # mu = [8.3, 19.9]
    # mu_time  = 0.3
    # ssig = 0.03
    # increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='RC')


    # ##########cc1
    # num_aft = int(450*7*3)
    # ccid = 1
    # size_file = 'Simulated_DataSets/Airplane/CC_size/CC1_size.csv'
    # # body_ssig_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    # # wing_ssig_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    # best_step = 1
    # gaussian_disribution_of_body_wing_CC(ccid, num_aft, size_file, best_step=best_step)

    ########### CC1 increase mu size 
    # num_aft = int(450*7*3)
    # common_cls = 1
    # mu = [28.6, 23.2]
    # mu_times  = [0.1, 0.2, 0.3]
    # # ssig = 0.08  # before 4/26/2021
    # ssig = np.array([0.08, 0.06])
    # gaussian_disribution_of_size_mu_CC(mu, ssig, mu_times, common_cls, num_aft)

    ####### increase sigma size based on large mu_size=mu*(1+0.3)
    # num_aft = int(450*7*3)
    # rare_cls = 1
    # print(f'-------------common {rare_cls}------------') 
    # mu = [28.6, 23.2]
    # mu_time  = 0.3
    # # ssig = 0.08  # before 4/26/2021
    # ssig = np.array([0.08, 0.06])
    # increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='CC')


    ######## CC2
    num_aft = int(450*7*3)
    ccid = 2
    best_step = 0
    size_file = 'Simulated_DataSets/Airplane/CC_size/CC2_size.csv'
    gaussian_disribution_of_body_wing_CC(ccid, num_aft, size_file, best_step=best_step)

    # ########### CC2 increase mu size  
    # # num_aft = int(450*7*3)
    # # common_cls = 2
    # # mu = [39.6, 34.1]
    # # mu_times  = [0.1, 0.2, 0.3]
    # # # ssig = 0 # before 4/26/2021
    # # ssig = np.array([0, 0])
    # # gaussian_disribution_of_size_mu_CC(mu, ssig, mu_times, common_cls, num_aft)
    
    # ####### increase sigma size based on large mu_size=mu*(1+0.3)
    # # num_aft = int(450*7*3)
    # rare_cls = 2
    # print(f'-------------common {rare_cls}------------')
    # mu = [39.6, 34.1]
    # mu_time  = 0.3
    # ssig = np.array([0, 0])
    # increase_sigma_size_based_on_optimal_ssig_with_large_mu_size(mu, mu_time, ssig, sig_size_list, rare_cls, num_aft, cmt='CC')

