from PIL import ImageColor
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

def gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft, wing_mu=None, cmt='RC'):
    np.random.seed(1)
    body_mu = np.array(body_mu)
    # save_dir = '../../result_output/RC_color/'
    save_dir = 'result_output/{}_color/'.format(cmt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if wing_mu:
        wing_txt = open(os.path.join(save_dir, '{}{}_wing_color.txt'.format(cmt, rare_cls)), 'w')
    body_txt = open(os.path.join(save_dir, '{}{}_body_color.txt'.format(cmt, rare_cls)), 'w')
    for ssig in ssig_list:
        diag_body = ssig * np.diag([1, 1, 1])
        # diag_body = ssig * np.diag([1, 1, 1])
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft) # convoriance
        # plt.hist(body_gs, 30)
        # plt.show()
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        for jx in range(body_gs.shape[0]):
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
        body_txt.write('@Hidden\nattr body_r{}= "{}"\n'.format(ssig, body_b))
        body_txt.write('@Hidden\nattr body_g{}= "{}"\n'.format(ssig, body_g))
        body_txt.write('@Hidden\nattr body_b{}= "{}"\n\n'.format(ssig, body_r))

        if wing_mu:
            diag_wing = ssig * np.diag([1, 1, 1])
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            for jx in range(wing_gs.shape[0]):
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])

            wing_txt.write('@Hidden\nattr wing_b{}= "{}"\n'.format(ssig, wing_r))
            wing_txt.write('@Hidden\nattr wing_g{}= "{}"\n'.format(ssig, wing_g))
            wing_txt.write('@Hidden\nattr wing_r{}= "{}"\n\n'.format(ssig, wing_b))
    body_txt.close()
    if wing_mu:
        wing_txt.close()


def gaussian_disribution_of_color_mu(body_ori, csig, mu_times, rare_cls, num_aft, wing_ori=None, cmt='RC'):
    #fixme 
    ########### before 4/21/2021 seed(1) is here
    np.random.seed(1)
    body_ori = np.array(body_ori)
    save_dir = 'result_output/{}_color_mu/'.format(cmt)
    # save_dir = '../../data_xview/1_cls/px23whr3_seed17/{}/{}_color/'.format(cmt, cmt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if wing_ori is not None:
        wing_txt = open(os.path.join(save_dir, '{}{}_wing_color.txt'.format(cmt, rare_cls)), 'w')
    body_txt = open(os.path.join(save_dir, '{}{}_body_color.txt'.format(cmt, rare_cls)), 'w')
    for ts in mu_times:
        diag_body = csig * np.diag([1, 1, 1])
        if rare_cls in [2, 3]:
            body_mu = body_ori*(1-ts)
        else:
            body_mu = body_ori*(1+ts)
        # body_mu = body_ori*(1+ts)
        print('body_mu', body_mu)
        # body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft) # convoriance
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft + 3000) # v2
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        cnt = 0
        for jx in range(body_gs.shape[0]):
            if np.any(body_gs[jx]>255) or np.any(body_gs[jx]<0): # v2
                continue
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
            cnt += 1
            if cnt == num_aft:
                break
        body_txt.write('@Hidden\nattr body_r_mu{}= "{}"\n'.format(int(ts*100), body_b))
        body_txt.write('@Hidden\nattr body_g_mu{}= "{}"\n'.format(int(ts*100), body_g))
        body_txt.write('@Hidden\nattr body_b_mu{}= "{}"\n\n'.format(int(ts*100), body_r))

        if wing_ori is not None:
            # if ts == 0.1:
            #     np.random.seed(1)
            diag_wing = csig * np.diag([1, 1, 1])
            wing_ori = np.array(wing_ori)
            if rare_cls in [2]: # only RC2 and CC2: the wing color is different from body
                wing_mu = wing_ori*(1-ts) 
            print('wing_mu', wing_mu)
            # wing_mu = wing_ori*(1+ts)
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft + 3000)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            ct = 0
            for jx in range(wing_gs.shape[0]):
                if np.any(wing_gs[jx]>255) or np.any(wing_gs[jx]<0): # v2
                    continue
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])
                ct += 1
                if ct == num_aft:
                    break

            wing_txt.write('@Hidden\nattr wing_b_mu{}= "{}"\n'.format(int(ts*100), wing_r))
            wing_txt.write('@Hidden\nattr wing_g_mu{}= "{}"\n'.format(int(ts*100), wing_g))
            wing_txt.write('@Hidden\nattr wing_r_mu{}= "{}"\n\n'.format(int(ts*100), wing_b))
    body_txt.close()
    plot_mu_dist(save_dir, '{}{}_body_color.txt'.format(cmt, rare_cls), body_ori)

    if wing_ori is not None:
        wing_txt.close()
        plot_mu_dist(save_dir, '{}{}_wing_color.txt'.format(cmt, rare_cls), wing_ori)


def gaussian_disribution_of_color_mu_CC(body_ori, csig, mu_times, common_cls, num_aft, wing_ori=None, wing_csig=None, cmt='CC'):
    np.random.seed(1)
    body_ori = np.array(body_ori)
    save_dir = 'result_output/CC_color_mu/'
    # save_dir = '../../data_xview/1_cls/px23whr3_seed17/{}/{}_color/'.format(cmt, cmt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if wing_ori is not None:
        wing_txt = open(os.path.join(save_dir, '{}{}_wing_color.txt'.format(cmt, common_cls)), 'w')
    body_txt = open(os.path.join(save_dir, '{}{}_body_color.txt'.format(cmt, common_cls)), 'w')
    for ts in mu_times:
        body_mu = body_ori*(1-ts)
        # body_mu = body_ori*(1+ts)
        print('body_mu', body_mu)
        diag_body = np.diag(np.power(csig*body_mu, 2))
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft) # convoriance
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        for jx in range(body_gs.shape[0]):
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
        body_txt.write('@Hidden\nattr body_r_mu{}= "{}"\n'.format(int(ts*100), body_b))
        body_txt.write('@Hidden\nattr body_g_mu{}= "{}"\n'.format(int(ts*100), body_g))
        body_txt.write('@Hidden\nattr body_b_mu{}= "{}"\n\n'.format(int(ts*100), body_r))

        if wing_ori is not None: # only for CC2
            wing_ori = np.array(wing_ori)
            # wing_mu = wing_ori*(1+ts) # before 4/21/2021 
            wing_mu = wing_ori*(1-ts)  # from 4/21/2021 
            diag_wing = np.diag(np.power(wing_csig*wing_mu, 2))
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            for jx in range(wing_gs.shape[0]):
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])

            wing_txt.write('@Hidden\nattr wing_b_mu{}= "{}"\n'.format(int(ts*100), wing_r))
            wing_txt.write('@Hidden\nattr wing_g_mu{}= "{}"\n'.format(int(ts*100), wing_g))
            wing_txt.write('@Hidden\nattr wing_r_mu{}= "{}"\n\n'.format(int(ts*100), wing_b))
    body_txt.close()
    if wing_ori is not None:
        wing_txt.close()


def compute_rgb_by_hex(hex_list):
    r_list = []
    g_list = []
    b_list = []
    for hex in hex_list:
        (r, g, b) = ImageColor.getcolor(hex, "RGB")
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
    r_mean = np.round(np.mean(r_list)).astype(np.int)
    g_mean = np.round(np.mean(g_list)).astype(np.int)
    b_mean = np.round(np.mean(b_list)).astype(np.int)
    r_std = np.around(np.std(r_list), decimals=2)
    g_std = np.around(np.std(g_list), decimals=2)
    b_std = np.around(np.std(b_list), decimals=2)
    rgb_mean = np.array([r_mean, g_mean, b_mean])
    rgb_std = np.array([r_std, g_std, b_std])
    print('mean', rgb_mean)
    print('std', rgb_std)
    return rgb_mean, rgb_std 


def gaussian_disribution_of_promu_rgb(body_mu, ssig_list, rare_cls, num_aft, wing_mu=None, wing_ssig_list=None, cmt='CC'):
    np.random.seed(1)
    body_mu = np.array(body_mu)
    save_dir = 'Simulated_DataSets/Airplane/{}_color/'.format(cmt)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(save_dir)
    if wing_mu is not None:
        wing_txt = open(os.path.join(save_dir, '{}{}_wing_color.txt'.format(cmt, rare_cls)), 'w')
    body_txt = open(os.path.join(save_dir, '{}{}_body_color.txt'.format(cmt, rare_cls)), 'w')
    for ssig in ssig_list:
        diag_body = np.diag(np.power(ssig*body_mu, 2))
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft) # convoriance
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        cnt = 0
        for jx in range(body_gs.shape[0]):
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
        body_txt.write('attr body_r{:g}= "{}"\n'.format(ssig[0]*100, body_b))
        body_txt.write('attr body_g{:g}= "{}"\n'.format(ssig[1]*100, body_g))
        body_txt.write('attr body_b{:g}= "{}"\n\n'.format(ssig[2]*100, body_r))
    body_txt.close()
    if wing_mu is not None:
        for ssig in wing_ssig_list:
            diag_wing = np.diag(np.power(ssig*wing_mu, 2))
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            for jx in range(wing_gs.shape[0]):
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])

            wing_txt.write('attr wing_r{:g}= "{}"\n'.format(ssig[0]*100, wing_r))
            wing_txt.write('attr wing_g{:g}= "{}"\n'.format(ssig[1]*100, wing_g))
            wing_txt.write('attr wing_b{:g}= "{}"\n\n'.format(ssig[2]*100, wing_b))

        wing_txt.close()


def gaussian_disribution_of_promu_rgb_no_boundary(body_mu, ssig_list, rare_cls, num_aft, wing_mu=None, wing_ssig_list=None, cmt='CC'):
    np.random.seed(1)
    body_mu = np.array(body_mu)
    save_dir = 'Simulated_DataSets/Airplane/{}_color/'.format(cmt)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(save_dir)
    if wing_mu is not None:
        wing_txt = open(os.path.join(save_dir, '{}{}_wing_color.txt'.format(cmt, rare_cls)), 'w')
    body_txt = open(os.path.join(save_dir, 'v2-{}{}_body_color.txt'.format(cmt, rare_cls)), 'w')
    for ssig in ssig_list:
        diag_body = np.diag(np.power(ssig, 2))# v2
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft + 3000) # v2
        body_b = ''
        body_g = ''
        body_r = ''
        cnt = 0
        for jx in range(body_gs.shape[0]):
            if np.any(body_gs[jx]>255) or np.any(body_gs[jx]<0): # v2
                continue
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
            cnt += 1
            if cnt == num_aft:
                break
        body_txt.write('attr body_r{:g}= "{}"\n'.format(ssig[0]*100, body_b))
        body_txt.write('attr body_g{:g}= "{}"\n'.format(ssig[1]*100, body_g))
        body_txt.write('attr body_b{:g}= "{}"\n\n'.format(ssig[2]*100, body_r))
    body_txt.close()
    if wing_mu is not None:
        for ssig in wing_ssig_list:
            diag_wing = np.diag(np.power(ssig*wing_mu, 2))
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft + 3000)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            for jx in range(wing_gs.shape[0]):
                if np.any(body_gs[jx]>255) or np.any(body_gs[jx]<0): # v2
                    continue
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])

            wing_txt.write('attr wing_r{:g}= "{}"\n'.format(ssig[0]*100, wing_r))
            wing_txt.write('attr wing_g{:g}= "{}"\n'.format(ssig[1]*100, wing_g))
            wing_txt.write('attr wing_b{:g}= "{}"\n\n'.format(ssig[2]*100, wing_b))

        wing_txt.close()


def increase_sigma_color_based_on_optimal_csig_with_large_mu_color(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori=None, wing_csig=None, cmt='RC'):
    '''
    for large_mu_color = (1+0.3)*mu_color
    '''
    np.random.seed(1)
    body_ori = np.array(body_ori)
    save_dir = 'result_output/{}_increased_color_sigma_based_on_optimal_csig/'.format(cmt)
    # save_dir = '../../data_xview/1_cls/px23whr3_seed17/{}/{}_color/'.format(cmt, cmt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if wing_ori is not None:
        wing_file_name = 'v6-{}{}_wing_color.txt'.format(cmt, rare_cls)
        wing_txt = open(os.path.join(save_dir, wing_file_name), 'w')
    body_file_name = 'v6-{}{}_body_color.txt'.format(cmt, rare_cls)
    body_txt = open(os.path.join(save_dir, body_file_name), 'w')

    if rare_cls in [2, 3]:
        body_mu = body_ori*(1-mu_time)
    else:
        body_mu = body_ori*(1+mu_time)
    print('body_mu', body_mu)
    for sc in sig_c_list: 
        # if not np.any(csig):
        #     diag_body = np.power(sc * 35, 2) * np.diag([1, 1, 1]) # v3
        #     # diag_body = np.power(sc * body_ori, 2) * np.diag([1, 1, 1])#v1
        # else:
        #     diag_body = np.power(csig * (1+sc), 2) * np.diag([1, 1, 1])
        std = sc * mu_time * body_ori
        print('std', std)
        diag_body = np.power(std, 2) * np.diag([1, 1, 1]) # v4

        print('diag_body', diag_body)
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft+ 3000) # convoriance
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        j = 0
        for jx in range(body_gs.shape[0]):
            if np.any(body_gs[jx]>255) or np.any(body_gs[jx]<0):
                continue
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
            j += 1
            if j == num_aft:
                break
        
        # lbl = round(float(sc),2) if sc <1 and sc >0 else int(sc) 
        lbl =  int(sc*100)
        body_txt.write('@Hidden\nattr body_r_mu{}= "{}"\n'.format(lbl, body_b))
        body_txt.write('@Hidden\nattr body_g_mu{}= "{}"\n'.format(lbl, body_g))
        body_txt.write('@Hidden\nattr body_b_mu{}= "{}"\n\n'.format(lbl, body_r))

        if wing_ori is not None: # only for CC2  and RC2
            wing_ori = np.array(wing_ori)
            if rare_cls in [2]:
                wing_mu = wing_ori*(1-mu_time)
            print('wing_mu', wing_mu)
            # if not np.any(wing_csig):
            #     diag_wing = np.power(sc * 35, 2) * np.diag([1, 1, 1])
            # else:
            #     diag_wing = np.power(wing_csig * (1+sc), 2) * np.diag([1,1,1])
            wing_std = sc * mu_time * wing_ori
            print('wing_std', wing_std)
            diag_wing = np.power(wing_std, 2) * np.diag([1, 1, 1]) # v4
            print('diag_wing', diag_wing)
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft + 3000)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            # wing_gs = np.clip(wing_gs, 0, 255)
            jj = 0
            for jx in range(wing_gs.shape[0]):
                if np.any(wing_gs[jx]>255) or np.any(wing_gs[jx]<0):
                    continue
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])
                jj += 1
                if jj == num_aft:
                    break

            # lbl = round(float(sc),2) if sc <1 and sc >0 else int(sc)
            lbl =  int(sc*100)
            wing_txt.write('@Hidden\nattr wing_b_mu{}= "{}"\n'.format(lbl, wing_r))
            wing_txt.write('@Hidden\nattr wing_g_mu{}= "{}"\n'.format(lbl, wing_g))
            wing_txt.write('@Hidden\nattr wing_r_mu{}= "{}"\n\n'.format(lbl, wing_b))
    body_txt.close()
    plot_sigma_dist(save_dir, body_file_name, body_ori, sig_c_list)
    
    if wing_ori is not None:
        wing_txt.close()
        plot_sigma_dist(save_dir, wing_file_name, wing_ori, sig_c_list)


def increase_sigma_color_based_on_optimal_csig_with_large_mu_color_for_CC(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori=None, wing_csig=None, cmt='CC'):
    '''
    for large_mu_color = (1+0.3)*mu_color
    '''
    np.random.seed(1) # before 4/24/2021
    body_ori = np.array(body_ori)
    save_dir = 'result_output/{}_increased_color_sigma_based_on_optimal_csig/'.format(cmt)
    # save_dir = '../../data_xview/1_cls/px23whr3_seed17/{}/{}_color/'.format(cmt, cmt)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if wing_ori is not None:
        wing_file_name = 'v6-{}{}_wing_color.txt'.format(cmt, rare_cls)
        wing_txt = open(os.path.join(save_dir, wing_file_name), 'w')
    body_file_name = 'v6-{}{}_body_color.txt'.format(cmt, rare_cls)
    body_txt = open(os.path.join(save_dir, body_file_name), 'w')
    
    body_mu = body_ori*(1-mu_time)
    print('body_mu', body_mu)
    for sc in sig_c_list: 
        # if not np.any(csig):
        #     sc_arr = np.array([sc*35 for _ in range(body_mu.shape[0])])
        #     diag_body = np.diag(np.power(sc_arr, 2))
        # else:
        #     sc_arr = np.array([1+sc for _ in range(body_mu.shape[0])])
        #     diag_body = np.diag(np.power(csig * sc_arr * body_mu, 2))
        
        std = sc * mu_time * body_ori
        print('std', std)
        diag_body = np.power(std, 2) * np.diag([1, 1, 1]) # v4
        
        print('diag_body', diag_body)
        body_gs = np.random.multivariate_normal(body_mu, diag_body, num_aft + 3000) # convoriance
        body_b = ''
        body_g = ''
        body_r = ''
        # body_gs = np.clip(body_gs, 0, 255)
        j = 0
        for jx in range(body_gs.shape[0]):
            if np.any(body_gs[jx]>255) or np.any(body_gs[jx]<0):
                continue
            body_b += '{:.1f};'.format(body_gs[jx, 0])
            body_g += '{:.1f};'.format(body_gs[jx, 1])
            body_r += '{:.1f};'.format(body_gs[jx, 2])
            j += 1
            if j == num_aft:
                break
        # lbl = round(float(sc),2) if sc <1 and sc >0 else int(sc) 
        lbl =  int(sc*100)
        body_txt.write('@Hidden\nattr body_r_mu{}= "{}"\n'.format(lbl, body_b))
        body_txt.write('@Hidden\nattr body_g_mu{}= "{}"\n'.format(lbl, body_g))
        body_txt.write('@Hidden\nattr body_b_mu{}= "{}"\n\n'.format(lbl, body_r))

        if wing_ori is not None: # only for CC2
            wing_ori = np.array(wing_ori)
            wing_mu = wing_ori*(1-mu_time) # from 4/21/2021
            print('wing_mu', wing_mu)
            # if not np.any(wing_csig):
            #     sc_arr = np.array([sc*35 for _ in range(wing_mu.shape[0])])
            #     diag_wing = np.diag(np.power(sc_arr, 2))
            # else:
            #     sc_arr = np.array([1+sc for _ in range(wing_mu.shape[0])])
            #     diag_wing = np.diag(np.power(wing_csig * sc_arr * wing_mu, 2))# v3 from 4/26/2021
            
            wing_std = sc * mu_time * wing_ori
        
            print('wing_std', wing_std)
            diag_wing = np.power(wing_std, 2) * np.diag([1, 1, 1]) # v4
            
            print('diag_wing', diag_wing)
            wing_gs = np.random.multivariate_normal(wing_mu, diag_wing, num_aft + 3000)
            wing_b = ''
            wing_g = ''
            wing_r = ''
            # wing_gs = np.clip(wing_gs, 0, 255)
            jj = 0
            for jx in range(wing_gs.shape[0]):
                if np.any(wing_gs[jx]>255) or np.any(wing_gs[jx]<0):
                    continue
                wing_b += '{:.1f};'.format(wing_gs[jx, 0])
                wing_g += '{:.1f};'.format(wing_gs[jx, 1])
                wing_r += '{:.1f};'.format(wing_gs[jx, 2])
                jj += 1
                if jj == num_aft:
                    break

            # lbl = round(float(sc),2) if sc <1 and sc >0 else int(sc)
            lbl =  int(sc*100)
            wing_txt.write('@Hidden\nattr wing_b_mu{}= "{}"\n'.format(lbl, wing_r))
            wing_txt.write('@Hidden\nattr wing_g_mu{}= "{}"\n'.format(lbl, wing_g))
            wing_txt.write('@Hidden\nattr wing_r_mu{}= "{}"\n\n'.format(lbl, wing_b))
    body_txt.close()
    plot_sigma_dist(save_dir, body_file_name, body_ori, sig_c_list)
    
    if wing_ori is not None:
        wing_txt.close()
        plot_sigma_dist(save_dir, wing_file_name, wing_ori, sig_c_list)


def plot_mu_dist(mu_path, mu_file, mu):
    df_rc = pd.read_csv(os.path.join(mu_path, mu_file), header=None)
    plt.figure()
    fig, axs = plt.subplots(1,3, sharey=True)

    color_list = ['r', 'g', 'b']
    for i in range(3):
        r = df_rc.loc[1+6*i, 0].split('=')[1]
        r = r.strip(' " ').split(';')[:-1]
        g = df_rc.loc[3+6*i, 0].split('=')[1]
        g = g.strip(' " ').split(';')[:-1]
        b = df_rc.loc[5+6*i, 0].split('=')[1]
        b = b.strip(' " ').split(';')[:-1]
        # print(r[-2:])
        r = [float(r) for r in r]
        g = [float(r) for r in g]
        b = [float(r) for r in b]
        # print('r', len(r))
        # print('g', len(g))
        # print('b', len(b))
        
        axs[0].hist(r, bins=50, color=color_list[i], alpha=0.5 , label=f"$\Delta \mu_r=$ {int(mu_times[i]*100)}") #
        axs[0].set_xlabel('R')
        axs[1].hist(g, bins=50, color=color_list[i], alpha=0.5 , label=f"$\Delta \mu_g=$ {int(mu_times[i]*100)}") #
        axs[1].set_xlabel('G')
        axs[2].hist(b, bins=50, color=color_list[i], alpha=0.5 , label=f"$\Delta \mu_b=$ {int(mu_times[i]*100)}") #
        axs[2].set_xlabel('B')
    plt.legend()
    title = mu_file.split('_color')[0]
    fig.suptitle(title)
    plt.savefig(os.path.join(mu_path, 'mu_' + mu_file.replace('.txt', '.png')))


def plot_sigma_dist(sigma_path, sigma_file, mu, sig_c_list):
    df_rc = pd.read_csv(os.path.join(sigma_path, sigma_file), header=None)
    plt.figure()
    fig, axs = plt.subplots(1,3, sharey=True)
    color_list = ['c', 'r', 'g', 'b']
    for i in range(1, 4):
        r = df_rc.loc[1+6*i, 0].split('=')[1]
        r = r.strip(' " ').split(';')[:-1]
        g = df_rc.loc[3+6*i, 0].split('=')[1]
        g = g.strip(' " ').split(';')[:-1]
        b = df_rc.loc[5+6*i, 0].split('=')[1]
        b = b.strip(' " ').split(';')[:-1]
        # print(r[-2:])
        r = [float(r) for r in r]
        g = [float(r) for r in g]
        b = [float(r) for r in b]
        
        rs_list = []
        gs_list = []
        bs_list = []
        for j in range(len(b)):
            # rs = np.abs(r[j]-mu[0])
            # gs = np.abs(g[j]-mu[1])
            # bs = np.abs(b[j]-mu[2])
            rs = (r[j]-mu[0])
            gs = (g[j]-mu[1])
            bs = (b[j]-mu[2])
            rs_list.append(rs)
            gs_list.append(gs)
            bs_list.append(bs)

        # lgd = f"$\Delta \sigma_c=$ {int(sig_c_list[i]*1)}"
        # lgd = f"$j=$ {int(sig_c_list[i]*1)}"
        lgd = "$j=$ {:.2f}".format(sig_c_list[i])
        axs[0].hist(rs_list, bins=50, color=color_list[i], alpha=0.4, label= lgd)
        axs[0].set_xlabel('R distance to $\mu_r$')
        axs[1].hist(gs_list, bins=50, color=color_list[i], alpha=0.4, label= lgd)
        axs[1].set_xlabel('G distance to $\mu_g$')
        axs[2].hist(bs_list, bins=50, color=color_list[i], alpha=0.4, label= lgd)
        axs[2].set_xlabel('B distance to $\mu_b$')
    plt.legend()
    title = sigma_file.split('_color')[0]
    fig.suptitle(title)
    plt.savefig(os.path.join(sigma_path, 'sigma_rgb_' + sigma_file.replace('.txt', '.png')))


if __name__ == '__main__':
    # np.random.seed(1)

    '''
    RC* color hex
    get RGB for each rc
    '''
    # RC1 rgb mean [130 115  95]
    # hexes = '#75705D;#726449;#836E5B;#766A5A;#AB9379;#8F745F;#7D7161;#807863;#7F725F'
    # RC2 body rgb mean [226 222 223]   wing rgb mean [174 171 170]
    # hexes = '#D4D0D1;#C5C2C9;#E5DAD6;#E9E5E6;#F0E5E1;#DEDFE3;#FBFCF6;#E6E4E5;#DCDDE1' # body
    # hexes = '#ABA6A3;#BCBBB7;#A7A7A5;#B3B2AD;#B6B1AE;#B4AEAE;#B4B2B3;#A9A4AA;#9A9591' #wing
    # RC3 rgb mean [249 241 232]
    # hexes = '#faf0e3;#f7f1e3;#fdfaee;#feecdf;#feefe5;#ecece4;#fefefb;#f0e1e7;#fef7e9'
    # RC4 rgb mean [65 65 56]
    # hexes = '#49484D;#404447;#4D4845;#434544;#4F4F4F;#3F4134;#373528;#464236'
    # hexes = '#403e31;#423f30;#373220;#393b2d;#433d2d;#42484e;#464c4b;#4b4c4f'
    # RC5 rgb mean [182 126  56]
    # hexes = '#BF853B;#9F6832;#B07C30;#D28F40;#926023;#C98834;#B2895D;#B07527;#CB8E3E'
    # seeds = np.random.choice(range(5000), 450)
    # hex_list = [s for s in hexes.split(';')]
    # r_list = []
    # g_list = []
    # b_list = []
    # for hex in hex_list:
    #     (r, g, b) = ImageColor.getcolor(hex, "RGB")
    #     r_list.append(r)
    #     g_list.append(g)
    #     b_list.append(b)
    # r_mean = np.round(np.mean(r_list)).astype(np.int)
    # g_mean = np.round(np.mean(g_list)).astype(np.int)
    # b_mean = np.round(np.mean(b_list)).astype(np.int)
    # rgb_mean = np.array([r_mean, g_mean, b_mean])
    # print('rgb mean', rgb_mean)

    '''
    gaussian dist values for each rc
    '''

    ########## RC1
    # rare_cls = 1
    # body_mu = [130, 115, 95]
    # num_aft = int(450*7*3)
    # # ssig_list = [0, 15, 30, 45, 60]
    # # ssig_list = [0, 5, 10, 15, 20]
    # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)

    # rare_cls = 1
    # num_aft = int(450*7*3)
    # body_mu =[130, 115, 95]
    # mu_times  = [0.1, 0.2, 0.3]
    # csig = 0
    # gaussian_disribution_of_color_mu(body_mu, csig, mu_times, rare_cls, num_aft)

    ################ sigma color
    # sig_c_list = np.array([0, 0.1, 0.2, 0.3], dtype=np.float32)*3 # v3
    # sig_c_list = np.array([0, 0.2, 0.4, 0.6], dtype=np.float32) # v2 
    # sig_c_list = np.array([0, 1, 2, 3], dtype=np.float32) # v4 v5
    # sig_c_list = np.array([0, 0.33, 0.66, 1], dtype=np.float32) # v6
    # num_aft_a = int(450*7*3)
    # num_aft = int(450*7*4)
    
    # ########## RC1
    ### mu_time = 0.3, with optima sigma_c then increse sigma_c by [0.1, 0.2, 0.3]
    # rare_cls = 1
    # print(f'-------------{rare_cls}------------')
    # # num_aft = int(450*7*3)
    # body_ori = [130, 115, 95]
    # mu_time = 0.3
    # csig = 0
    # increase_sigma_color_based_on_optimal_csig_with_large_mu_color(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori=None, cmt='RC')

    
    # # ########## RC2
    # # rare_cls = 2
    # # body_mu = [226, 222, 223]
    # # wing_mu = [174, 171, 170]
    # # num_aft = int(450*7*3)
    # # # ssig_list = [0, 15, 30, 45, 60]
    # # # ssig_list = [0, 5, 10, 15, 20]
    # # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft, wing_mu)

    # rare_cls = 2
    # num_aft = int(450*7*3)
    # body_mu = [226, 222, 223]
    # wing_mu = [174, 171, 170]
    # mu_times  = [0.1, 0.2, 0.3]
    # csig = 10**2
    # gaussian_disribution_of_color_mu(body_mu, csig, mu_times, rare_cls, num_aft, wing_ori=wing_mu)
    
    # #### mu_time = 0.3, with optima sigma_c then increse sigma_c by [0.1, 0.2, 0.3]
    # rare_cls = 2
    # print(f'-------------{rare_cls}------------')
    # # num_aft = int(450*7*3)
    # body_ori = [226, 222, 223]
    # wing_ori = [174, 171, 170]
    # mu_time = 0.3
    # csig = 10
    # increase_sigma_color_based_on_optimal_csig_with_large_mu_color(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori, cmt='RC')

    
    # # ########## RC3
    # # rare_cls = 3
    # # body_mu =[249, 241, 232]
    # # num_aft = int(450*7*3)
    # # # ssig_list = [0, 15, 30, 45, 60]
    # # # ssig_list = [0, 5, 10, 15, 20]
    # # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)

    # # rare_cls = 3
    # # num_aft = int(450*7*3)
    # # body_mu =[249, 241, 232]
    # # mu_times  = [0.1, 0.2, 0.3]
    # # csig = 40**2
    # # gaussian_disribution_of_color_mu(body_mu, csig, mu_times, rare_cls, num_aft)

    # ### mu_time = 0.3, with optima sigma_c then increse sigma_c by [0.1, 0.2, 0.3]
    # rare_cls = 3
    # print(f'-------------{rare_cls}------------')
    # # num_aft = int(450*7*3)
    # body_ori = [249, 241, 232]
    # mu_time = 0.3
    # csig = 40
    # increase_sigma_color_based_on_optimal_csig_with_large_mu_color(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori=None, cmt='RC')

    
    # ########## RC4
    # # rare_cls = 4
    # # body_mu =[65, 65, 56]
    # # num_aft = int(450*7*3)
    # # # ssig_list = [0, 15, 30, 45, 60]
    # # # ssig_list = [0, 5, 10, 15, 20]
    # # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)

    # rare_cls = 4
    # body_mu =[65, 65, 56]
    # num_aft = int(450*7*3)
    # mu_times  = [0.1, 0.2, 0.3]
    # csig = 0
    # gaussian_disribution_of_color_mu(body_mu, csig, mu_times, rare_cls, num_aft) 

    # ### mu_time = 0.3, with optima sigma_c then increse sigma_c by [0.1, 0.2, 0.3]
    # rare_cls = 4
    # print(f'-------------{rare_cls}------------')
    # # num_aft = int(450*7*3)
    # body_ori = [65, 65, 56]
    # mu_time = 0.3
    # csig = 0
    # increase_sigma_color_based_on_optimal_csig_with_large_mu_color(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori=None, cmt='RC')


    # # ########## RC5
    # # rare_cls = 5
    # # body_mu =[182, 126, 56]
    # # num_aft = int(450*7*3)
    # # # ssig_list = [0, 15, 30, 45, 60]
    # # # ssig_list = [0, 5, 10, 15, 20]
    # # ssig_list = [0, 10**2, 20**2, 30**2, 40**2]
    # # gaussian_disribution_of_color(body_mu, ssig_list, rare_cls, num_aft)

    # # rare_cls = 5
    # # num_aft = int(450*7*3)
    # # body_mu =[182, 126, 56]
    # # mu_times  = [0.1, 0.2, 0.3]
    # # csig = 0
    # # gaussian_disribution_of_color_mu(body_mu, csig, mu_times, rare_cls, num_aft)

    # #### mu_time = 0.3, with optima sigma_c then increse sigma_c by [0.1, 0.2, 0.3]
    # rare_cls = 5
    # print(f'-------------{rare_cls}------------')
    # # num_aft = int(450*7*3)
    # body_ori = [182, 126, 56]
    # mu_time = 0.3
    # csig = 0
    # increase_sigma_color_based_on_optimal_csig_with_large_mu_color(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori=None, cmt='RC')


    ########## CC1
    # ccid = 1
    # num_aft = int(450*7*3)
    # # num_aft_a = int(450*7*4)
    # hex_file = 'Simulated_DataSets/Airplane/CC_color/CC1_hex.csv'
    # df_hex = pd.read_csv(hex_file)
    # hex_list = df_hex['Body'].tolist()
    # rgb_mean, rgb_std = compute_rgb_by_hex(hex_list)
    # step_body = np.around(rgb_std/rgb_mean/4*0.9, decimals=2)
    # best_step = 1
    # body_ssig_list = [step_body*best_step]

    # print('step_body',step_body)
    # print('body_ssig_list', body_ssig_list)
    # # gaussian_disribution_of_promu_rgb(rgb_mean, body_ssig_list, ccid, num_aft,  cmt='CC')
    # gaussian_disribution_of_promu_rgb_no_boundary(rgb_mean, body_ssig_list, ccid, num_aft,  cmt='CC')

    # # common_cls = 1
    # # num_aft = int(450*7*3)
    # # body_mu = [199, 190, 182]
    # # mu_times  = [0.1, 0.2, 0.3]
    # # csig = np.array([0.05, 0.05, 0.06])*1
    # # gaussian_disribution_of_color_mu_CC(body_mu, csig, mu_times, common_cls, num_aft)

    # #### mu_time = 0.3, with optima sigma_c then increse sigma_c by [0.1, 0.2, 0.3]
    # rare_cls = 1
    # print(f'-------------common {rare_cls}------------')
    # # num_aft = int(450*7*3)
    # body_ori = [199, 190, 182]
    # mu_time = 0.3
    # csig = np.array([0.05, 0.05, 0.06])*1
    # increase_sigma_color_based_on_optimal_csig_with_large_mu_color_for_CC(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, cmt='CC')


    ########## CC2
    ccid = 2
    num_aft = int(450*7*3)
    hex_file = 'Simulated_DataSets/Airplane/CC_color/CC2_hex.csv'
    # hex_file = 'result_output/CC_color/CC2_new_hex.csv'
    df_hex = pd.read_csv(hex_file, sep='\t')
    body_hex_list = df_hex['Body'].tolist()
    rgb_mean, rgb_std = compute_rgb_by_hex(body_hex_list)
    step_body = np.around(rgb_std/rgb_mean/4*0.9, decimals=2)
    best_step = 3
    body_ssig_list = [best_step*step_body]
    print('step_body',step_body)
    print('body_ssig_list', body_ssig_list)
    
    wing_hex_list = df_hex['Wing'].tolist()
    wing_rgb_mean, wing_rgb_std = compute_rgb_by_hex(wing_hex_list)
    step_wing = np.around(wing_rgb_std/wing_rgb_mean/4*0.9, decimals=2)
    wing_ssig_list = [best_step*step_wing]
    print('step_wing',step_wing)
    print('wing_ssig_list', wing_ssig_list)
    gaussian_disribution_of_promu_rgb(rgb_mean, body_ssig_list, ccid, num_aft, wing_mu=wing_rgb_mean, wing_ssig_list=wing_ssig_list, cmt='CC')

    # common_cls = 2
    # num_aft = int(450*7*3)
    # body_mu = [230, 218, 213]
    # wing_mu = [176, 172, 164]
    # mu_times  = [0.1, 0.2, 0.3]
    # #### the first version forgot to multipy 3
    # # csig =np.array( [0.04, 0.05, 0.04])
    # # wing_csig = np.array([0.04, 0.05, 0.05])
    # csig =np.array( [0.04, 0.05, 0.04])*3
    # wing_csig = np.array([0.04, 0.05, 0.05])*3
    # gaussian_disribution_of_color_mu_CC(body_mu, csig, mu_times, common_cls, num_aft, wing_ori=wing_mu, wing_csig=wing_csig)

    # #### mu_time = 0.3, with optima sigma_c then increse sigma_c by [0.1, 0.2, 0.3]
    # rare_cls = 2
    # print(f'-------------common {rare_cls}------------')
    # # num_aft = int(450*7*3)
    # body_ori = [230, 218, 213]
    # wing_ori = [176, 172, 164]
    # mu_time = 0.3
    # csig = np.array([0.04, 0.05, 0.04])*3
    # print('csig', csig)
    # wing_csig = np.array([0.04, 0.05, 0.05])*3
    # print('wing_csig', wing_csig)
    # increase_sigma_color_based_on_optimal_csig_with_large_mu_color_for_CC(body_ori, mu_time, csig, sig_c_list, rare_cls, num_aft, wing_ori, wing_csig, cmt='CC')






    
    