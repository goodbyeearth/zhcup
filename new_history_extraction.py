# 提取历史跟label有关的feature
import pandas as pd
import logging
import gc
import os
import pickle
from sklearn.preprocessing import LabelEncoder

log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
logging.basicConfig(format=log_fmt, level=logging.INFO)

base_path = '../biendata_kanshanbei/data'
feature_path = './feature'
feature_voygern_train = f'{feature_path}/train_voygern_feature.txt'
feature_voygern_test = f'{feature_path}/test_voygern_feature.txt'
encoder_dic_file = f'{base_path}/enc_dic.pkl'

def load_invite_info():
    train = pd.read_csv(f'{base_path}/invite_info_0926.txt', sep='\t', header=None)
    train.columns = ['qid', 'uid', 'dt', 'label']
    logging.info("invite %s", train.shape)

    test = pd.read_csv(f'{base_path}/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
    test.columns = ['qid', 'uid', 'dt']
    logging.info("test %s", test.shape)

    data = pd.concat([train,test]).reset_index(drop=True)
    data['day'] = data['dt'].apply(lambda x:int(x.split('-')[0].split('D')[1]))
    data['hour'] = data['dt'].apply(lambda x:int(x.split('-')[1].split('H')[1]))
    data['wk'] = data['day'] % 7
    
    del data['dt']
    
   

    # 加载问题信息
    ques = pd.read_csv(f'{base_path}/question_info_0926.txt', header=None, sep='\t')
    ques.columns =  ['qid','create_time','title','title_cut','descrip','descrip_cut','bound_topic']
    ques['q_day'] = ques.create_time.apply(lambda x: int(x.split('-')[0].split('D')[1]))
    ques['q_hour'] = ques.create_time.apply(lambda x: int(x.split('-')[1].split('H')[1]))
    # print(ques['qid','q_day','q_hour'].head(5))
    data = pd.merge(data,ques[['qid','q_day','q_hour']],on='qid',how='left')
    del ques
    gc.collect()
    data['diff_iq_day'] = data['day'] - data['q_day']
    data['diff_iq_hour'] = data['day']*24 + data['hour'] - data['q_day']*24 - data['q_hour']

    # 加载用户
    user = pd.read_csv(f'{base_path}/member_info_0926.txt', header=None, sep='\t')
    user.columns = ['uid', 'gender' ,'keyword','volume_level','heat_level','reg_type','reg_stage','freq',
                    'uf_b1', 'uf_b2','uf_b3', 'uf_b4', 'uf_b5', 
                    'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 
                    'score', 'follow_topic', 'inter_topic']
    
    drop_feat = ['keyword','volume_level','heat_level','reg_type','reg_stage']
    user.drop(drop_feat,axis=1,inplace=True)
    logging.info("user %s", user.shape)
    class_feat =  ['ffa','ffb','ffc','ffd','ffe','sex','freq']
    user_feat = ['uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5','gender','freq']
    user_feat_dict = {user_feat[i]:class_feat[i] for i in range(len(user_feat))}
   
    
    data = pd.merge(data, user, on='uid', how='left')
    enc_dic = pickle.load(open(encoder_dic_file,'rb'))
    for feat in user_feat:
        data[feat] = enc_dic[user_feat_dict[feat]].transform(data[feat])
    del user
    gc.collect()

    # 加载qu_topic_count,qu_topic_count_weight
    t1 = pd.read_csv(f'./feature/train_kfold_topic_feature.txt', sep='\t', 
                 usecols=['qu_topic_count_weight', 'qu_topic_count'])
    

    t2 = pd.read_csv(f'./feature/test_kfold_topic_feature.txt', sep='\t', 
                    usecols=['qu_topic_count_weight', 'qu_topic_count'])
    t = pd.concat([t1,t2]).reset_index(drop=True)
    print('tshape: ',t.shape)
    print(data.shape)
    data = pd.concat([data,t],axis=1)

  
    
    t1 = pd.read_csv(f'./feature/train_invite_feature_2.txt', sep='\t', 
                 usecols=['intersection_ft_count', 'intersection_it_count'])
    

    t2 = pd.read_csv(f'./feature/test_invite_feature_2.txt', sep='\t', 
                    usecols=['intersection_ft_count', 'intersection_it_count'])
    t = pd.concat([t1,t2]).reset_index(drop=True)
    data = pd.concat([data,t],axis=1)

    return len(train),data


def load_invite_info_test2():
    train = pd.read_csv(f'{base_path}/invite_info_0926.txt', sep='\t', header=None)
    train.columns = ['qid', 'uid', 'dt', 'label']
    logging.info("invite %s", train.shape)

    test = pd.read_csv(f'{base_path}/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
    test.columns = ['qid', 'uid', 'dt']
    logging.info("test %s", test.shape)

    test1 = pd.read_csv(f'{base_path}/invite_info_evaluate_2_0926.txt', sep='\t', header=None)
    test1.columns = ['qid', 'uid', 'dt']
    logging.info("test %s", test1.shape)

    data = pd.concat([train,test,test1]).reset_index(drop=True)
    data['day'] = data['dt'].apply(lambda x:int(x.split('-')[0].split('D')[1]))
    data['hour'] = data['dt'].apply(lambda x:int(x.split('-')[1].split('H')[1]))
    data['wk'] = data['day'] % 7
    
    del data['dt']
    
   

    # 加载问题信息
    ques = pd.read_csv(f'{base_path}/question_info_0926.txt', header=None, sep='\t')
    ques.columns =  ['qid','create_time','title','title_cut','descrip','descrip_cut','bound_topic']
    ques['q_day'] = ques.create_time.apply(lambda x: int(x.split('-')[0].split('D')[1]))
    ques['q_hour'] = ques.create_time.apply(lambda x: int(x.split('-')[1].split('H')[1]))
    # print(ques['qid','q_day','q_hour'].head(5))
    data = pd.merge(data,ques[['qid','q_day','q_hour']],on='qid',how='left')
    del ques
    gc.collect()
    data['diff_iq_day'] = data['day'] - data['q_day']
    data['diff_iq_hour'] = data['day']*24 + data['hour'] - data['q_day']*24 - data['q_hour']

    # 加载用户
    user = pd.read_csv(f'{base_path}/member_info_0926.txt', header=None, sep='\t')
    user.columns = ['uid', 'gender' ,'keyword','volume_level','heat_level','reg_type','reg_stage','freq',
                    'uf_b1', 'uf_b2','uf_b3', 'uf_b4', 'uf_b5', 
                    'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 
                    'score', 'follow_topic', 'inter_topic']
    
    drop_feat = ['keyword','volume_level','heat_level','reg_type','reg_stage']
    user.drop(drop_feat,axis=1,inplace=True)
    logging.info("user %s", user.shape)
    class_feat =  ['ffa','ffb','ffc','ffd','ffe','sex','freq']
    user_feat = ['uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5','gender','freq']
    user_feat_dict = {user_feat[i]:class_feat[i] for i in range(len(user_feat))}
   
    
    data = pd.merge(data, user, on='uid', how='left')
    enc_dic = pickle.load(open(encoder_dic_file,'rb'))
    for feat in user_feat:
        # data[feat] = enc_dic[user_feat_dict[feat]].transform(data[feat])
        lb = LabelEncoder()
        lb.fit(data[feat])
        data[feat] = lb.transform(data[feat])
    del user
    gc.collect()

    # 加载qu_topic_count,qu_topic_count_weight 本次跑没有embedding信息
    t1 = pd.read_csv(f'./feature/train_kfold_topic_feature.txt', sep='\t', 
                 usecols=['qu_topic_count_weight', 'qu_topic_count'])
    

    t2 = pd.read_csv(f'./feature/test_kfold_topic_feature.txt', sep='\t', 
                    usecols=['qu_topic_count_weight', 'qu_topic_count'])

    t3 = pd.read_csv(f'./feature/newtest_kfold_topic_feature.txt', sep='\t', 
                    usecols=['qu_topic_count_weight', 'qu_topic_count'])
    t = pd.concat([t1,t2,t3]).reset_index(drop=True)
    print('tshape: ',t.shape)
    print(data.shape)
    data = pd.concat([data,t],axis=1)

  
    
    t1 = pd.read_csv(f'./feature/train_invite_feature_2.txt', sep='\t', 
                 usecols=['intersection_ft_count', 'intersection_it_count'])
    

    t2 = pd.read_csv(f'./feature/test_invite_feature_2.txt', sep='\t', 
                    usecols=['intersection_ft_count', 'intersection_it_count'])

    t3 = pd.read_csv(f'./feature/test_invite_feature_2.txt', sep='\t', 
                    usecols=['intersection_ft_count', 'intersection_it_count'])
    t = pd.concat([t1,t2,t3]).reset_index(drop=True)

    data = pd.concat([data,t],axis=1)

    return len(train),len(train)+len(test),data

def group_ops(data,gp,feat,ops,alias=''):
    t = data.groupby(gp)[feat].agg(ops).reset_index()
    if type(gp) is not list:
        gp = [gp]
    if type(ops) is list:
        t.columns = gp + [alias+'_'.join(gp)+'__'+'_'.join(x) for x in t.columns.ravel() if x[0] not in gp]
    else:
        t.columns = gp + [alias+'_'.join(gp)+'__'+'_'.join(x) for x in t.columns if x not in gp]
    return t

# last week : d-13~d-6            13
# [{uid,qid}]_{label}_{mean,std,count,sum}  8
# [qid_{user_feat}]_label_{mean,std,count,sum}  52
# [{uid,qid}_{diff_iq_day,qu_topic_count,qu_topic_count_weight}]_{label}_{mean,std,count,sum} 24
# [uid_{wk,hour}]_label_{mean,std,count,sum} 8
# [{intersection,userfeats,diff_iq_hour,diff_iq_day}]_label_{mean,std,count,sum} 68
#  = 13+8+52+24+8+68 = 73+32+68 = 173
def generate_groups():
    gps = []
    user_feats = ['uf_b1', 'uf_b2','uf_b3', 'uf_b4', 'uf_b5', 
         'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 
         'score', 'freq', 'gender']

    # gps.append(user_feats)
    # gps.append(user_feats[:5])
    # gps.append(user_feats[5:10])
    gps.append(['uid'])
    gps.append(['qid'])
    for x in ['wk','hour']:
        gps.append(['uid',x])
    
    for x in ['diff_iq_day','qu_topic_count','qu_topic_count_weight']:
        gps.append(['qid',x])
        gps.append(['uid',x])

    for x in user_feats:
        gps.append(['qid',x])
        gps.append([x])

    for x in ['intersection_ft_count','intersection_ft_count','diff_iq_hour','diff_iq_day']:
        gps.append([x])

    return gps

def extract_lw_feature(data,mode=''):
    res= []
    gps = generate_groups()
    for i in range(len(gps)):
        res.append([])

    ops = ['mean','std','count','sum']
    print('last week ', mode)
    if mode == 'test2':
        start,end = 3868,3875
    else:
        start,end = 3851,3875

    for d in range(start,end):
        print('day ',d)
        sel = data[(data.day >= d-13)&(data.day < d-6)]
        gc.collect()
        for i in range(len(gps)):
            t = group_ops(sel,gps[i],['label'],ops)
            t['day'] = d
            res[i].append(t)
    
    # merge on data
    print('start to merge')
    for i in range(len(gps)):
        print('working on ',i)
        data = data.merge(pd.concat(res[i]),on=['day']+gps[i],how='left')
    
    return data


def extract_h6_feature(data,mode=''):
    res= []
    gps = generate_groups()
    for i in range(len(gps)):
        res.append([])

    ops = ['mean','std','count','sum']
    print('less than 6d ',mode)
    if mode == 'test2':
        start,end = 3868,3875
    else:
        start,end = 3844,3875

    for d in range(start,end):
        print('day ',d)
        sel = data[(data.day < d-6)]
        window_size = d-6-3838
        if len(sel) == 0:
            print(f'day {d} windowsize = 0')
            continue
        else:
            print(f'day {d} windowsize = {window_size}')
            
        gc.collect()
        for i in range(len(gps)):
            t = group_ops(sel,gps[i],['label'],ops)
            t['day'] = d
            for x in t.columns:
                if 'count' in x:
                    t[x] /= window_size
            res[i].append(t)

    # merge on data
    print('start to merge')
    for i in range(len(gps)):
        print('working on ',i)
        data = data.merge(pd.concat(res[i]),on=['day']+gps[i],how='left')

    return data

def extract_hd_feature(data,mode=''):
    res= []
    gps = generate_groups()
    for i in range(len(gps)):
        res.append([])

    ops = ['mean','std','count','sum']

    print('less than d ',mode)
    if mode == 'test2':
        start,end = 3868,3875
    else:
        start,end = 3839,3875

    for d in range(start,end):
        print('day ',d)
        sel = data[(data.day < d)]
        if d <= 3867 :
            window_size = d-3838
        else:
            window_size = 3867-3838
        if len(sel) == 0:
            print(f'day {d} windowsize = 0')
            continue
        else:
            print(f'day {d} windowsize = {window_size}')
        gc.collect()
        for i in range(len(gps)):
            t = group_ops(sel,gps[i],['label'],ops)
            t['day'] = d
            for x in t.columns:
                if 'count' in x:
                    t[x] /= window_size
            res[i].append(t)
    
    # merge on data
    print('start to merge')
    for i in range(len(gps)):
        print('working on ',i)
        data = data.merge(pd.concat(res[i]),on=['day']+gps[i],how='left')   

    return data    
    

def extract_all():
    # ltrain,data = load_invite_info()
    # original_cols = list(data.columns)
    # data = extract_lw_feature(data)
    # after_cols = list(data.columns)
    # data.drop(original_cols,axis=1,inplace=True)
    # pickle.dump(data,open('./feature/new_history_lastweek.pkl','wb'),protocol=4)
    # del data
    # gc.collect()

    ltrain,data = load_invite_info()
    original_cols = list(data.columns)
    data = extract_h6_feature(data)
    after_cols = list(data.columns)
    data.drop(original_cols,axis=1,inplace=True)
    pickle.dump(data,open('./feature/new_history_ltd6.pkl','wb'),protocol=4)
    del data
    gc.collect()

    ltrain,data = load_invite_info()
    original_cols = list(data.columns)
    data = extract_hd_feature(data)
    after_cols = list(data.columns)
    data.drop(original_cols,axis=1,inplace=True)
    pickle.dump(data,open('./feature/new_history_ltd.pkl','wb'),protocol=4)

def extract_all_test2():
    ltrain,ltest,data = load_invite_info_test2()
    original_cols = list(data.columns)
    data = extract_lw_feature(data,'test2')
    after_cols = list(data.columns)
    data.drop(original_cols,axis=1,inplace=True)
    pickle.dump(data.iloc[ltest:],open('./feature/new_history_lastweek_test2.pkl','wb'),protocol=4)
    del data
    gc.collect()

    ltrain,ltest,data = load_invite_info_test2()
    original_cols = list(data.columns)
    data = extract_h6_feature(data,'test2')
    after_cols = list(data.columns)
    data.drop(original_cols,axis=1,inplace=True)
    pickle.dump(data.iloc[ltest:],open('./feature/new_history_ltd6_test2.pkl','wb'),protocol=4)
    del data
    gc.collect()

    ltrain,ltest,data = load_invite_info_test2()
    original_cols = list(data.columns)
    data = extract_hd_feature(data,'test2')
    after_cols = list(data.columns)
    data.drop(original_cols,axis=1,inplace=True)
    pickle.dump(data.iloc[ltest:],open('./feature/new_history_ltd_test2.pkl','wb'),protocol=4)

extract_all_test2()