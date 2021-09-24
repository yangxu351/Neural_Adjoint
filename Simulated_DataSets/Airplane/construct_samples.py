import pandas as pd
import os


if __name__ == '__main__':
    size_dir = 'Simulated_DataSets/Airplane/CC_size'
    color_dir = 'Simulated_DataSets/Airplane/CC_color'
    save_dir = 'Simulated_DataSets/Airplane' 
    
    #############################
    #######  CC1          #######
    #############################
    # ccid = 1
    # save_name = f'CC{ccid}_sample_list.csv'
    # size_file_name = f'syn_CC{ccid}_size.txt'
    # body_ssig_list = [0.08*100]
    # wing_ssig_list = [0.06*100]
    # body_color_file_name = f'v2-CC{ccid}_body_color.txt'
    # wing_color_file_name = f'v2-CC{ccid}_body_color.txt'
    # best_step = 1
    # body_csig_list = [int(i*best_step*100) for i in [0.05, 0.05, 0.06]]

    #############################
    #######  CC2          #######
    #############################
    ccid = 2
    save_name = f'CC{ccid}_sample_list.csv'
    size_file_name = f'syn_CC{ccid}_size.txt'
    body_ssig_list = [0]
    wing_ssig_list = [0]
    body_color_file_name = f'CC{ccid}_body_color.txt'
    wing_color_file_name = f'CC{ccid}_wing_color.txt'
    best_step = 3
    body_csig_list = [int(i*best_step*100) for i in [0.04, 0.05, 0.04]]
    wing_csig_list = [int(i*best_step*100) for i in [0.04, 0.05, 0.05]]
    
    #############################
    ####### Construct     #######
    #############################
    df_samples = pd.DataFrame(columns=['Body_size', 'Wing_size', 'Body_R', 'Body_G', 'Body_B', 'Wing_R', 'Wing_G', 'Wing_B'])
    
    df_size = pd.read_table(os.path.join(size_dir, size_file_name), sep='\n', header=None)
    print(df_size.shape)
    str_size = df_size.iloc[0, 0].split('"')[1]
    str_size = str_size.split(';')[:-1]
    size_list = [float(i) for i in str_size]
    df_samples['Body_size'] = size_list

    str_size = df_size.iloc[1, 0].split('"')[1]
    str_size = str_size.split(';')[:-1]
    size_list = [float(i) for i in str_size]
    df_samples['Wing_size'] = size_list

    df_color = pd.read_table(os.path.join(color_dir, body_color_file_name), sep='\n', header=None)
    print(df_color.shape)
    r_color = df_color.iloc[0, 0].split('"')[1]
    r_color = r_color.split(';')[:-1]
    r_list = [float(i) for i in r_color]
    df_samples['Body_R'] = r_list

    g_color = df_color.iloc[1, 0].split('"')[1]
    g_color = g_color.split(';')[:-1]
    g_list = [float(i) for i in g_color]
    df_samples['Body_G'] = g_list

    b_color = df_color.iloc[2, 0].split('"')[1]
    b_color = b_color.split(';')[:-1]
    b_list = [float(i) for i in b_color]
    df_samples['Body_B'] = b_list

    df_color = pd.read_table(os.path.join(color_dir, wing_color_file_name), sep='\n', header=None)
    print(df_color.shape)
    r_color = df_color.iloc[0, 0].split('"')[1]
    r_color = r_color.split(';')[:-1]
    r_list = [float(i) for i in r_color]
    df_samples['Wing_R'] = r_list

    g_color = df_color.iloc[1, 0].split('"')[1]
    g_color = g_color.split(';')[:-1]
    g_list = [float(i) for i in g_color]
    df_samples['Wing_G'] = g_list

    b_color = df_color.iloc[2, 0].split('"')[1]
    b_color = b_color.split(';')[:-1]
    b_list = [float(i) for i in b_color]
    df_samples['Wing_B'] = b_list

    df_samples.to_csv(os.path.join(save_dir, save_name), index=False)
    
