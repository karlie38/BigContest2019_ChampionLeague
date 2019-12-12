import numpy as np
import pandas as pd

class DataSet:
    def __init__(self):
        self.train_label = pd.read_csv('../raw/train_label.csv')
        self.train_activity = pd.read_csv('../raw/train_activity.csv')
        self.train_pay = pd.read_csv('../raw/train_payment.csv')
        self.train_trade = pd.read_csv('../raw/train_trade.csv')
        self.train_pled = pd.read_csv('../raw/train_pledge.csv')
        self.train_combat = pd.read_csv('../raw/train_combat.csv')
        
        self.test1_activity = pd.read_csv('../raw/test1_activity.csv')
        self.test1_pay = pd.read_csv('../raw/test1_payment.csv')
        self.test1_trade = pd.read_csv('../raw/test1_trade.csv')
        self.test1_pled = pd.read_csv('../raw/test1_pledge.csv')
        self.test1_combat = pd.read_csv('../raw/test1_combat.csv')
        
        self.test2_activity = pd.read_csv('../raw/test2_activity.csv')
        self.test2_pay = pd.read_csv('../raw/test2_payment.csv')
        self.test2_trade = pd.read_csv('../raw/test2_trade.csv')
        self.test2_pled = pd.read_csv('../raw/test2_pledge.csv')
        self.test2_combat = pd.read_csv('../raw/test2_combat.csv')
    
    def get_train_data(self):
        return self.train_label, self.train_activity, self.train_pay,\
               self.train_trade, self.train_pled, self.train_combat

    def get_test1_data(self):
        return self.test1_activity, self.test1_pay,\
               self.test1_trade, self.test1_pled, self.test1_combat
    
    def get_test2_data(self):
        return self.test2_activity, self.test2_pay,\
               self.test2_trade, self.test2_pled, self.test2_combat

def survival_time_preprocessing(activity, pay, trade, pled, combat):
    #하루 평균 playtime, npc_kill, solo_exp, party_exp, quest_exp, fishing, pricate_shop, game_money_shop
    #exp_recovery, enchant_count
    aa = activity.pivot_table(index = ['day', 'acc_id'], aggfunc='mean',
                              values = ['playtime','npc_kill','solo_exp', 'party_exp', 'quest_exp', 'fishing',
                                        'private_shop','game_money_change', 'exp_recovery', 'enchant_count']).reset_index()
    # 하루 평균 캐릭터 수
    aa2 = activity.pivot_table(index = ['day', 'acc_id'], aggfunc='count',
                               values = 'char_id').reset_index()
                                        
    # 하루 평균 결제 금액
    aa3 = pay.pivot_table(index = ['day', 'acc_id'], aggfunc='mean', values = 'amount_spent' ).reset_index()
    aa3 = aa3.rename(columns = {'amount_spent': 'pay_mean'})
                                        
    # 하루 총 결제 금액
    pp2 = pay.pivot_table(index=['day', 'acc_id'], aggfunc='sum', values='amount_spent').fillna(0).reset_index()
    pp2 = pp2.rename(columns = {'amount_spent': 'pay_sum'})
                                        
    # 하루 총 결제 횟수
    pp1 = pay.pivot_table(index = ['day', 'acc_id'], aggfunc='count', values = 'amount_spent' ).reset_index()
    pp1 = pp1.rename(columns = {'amount_spent': 'pay_count'})
                                        
    # 거래에서 하루 평균 buy_price, sell_price, 판매 대비 구매 비율
    aa4 = trade.pivot_table(index = ['day', 'source_acc_id'], values = 'item_price', aggfunc='mean').reset_index()
    aa4.rename(columns = {'source_acc_id': 'acc_id', 'item_price': 'sell_price'}, inplace=True)
    aa5 = trade.pivot_table(index = ['day','target_acc_id'], values = 'item_price', aggfunc='mean').reset_index()
    aa5.rename(columns = {'target_acc_id': 'acc_id', 'item_price': 'buy_price'}, inplace=True)
    판매구매금액 = pd.merge(aa4, aa5, on=('day','acc_id'), how='outer').fillna(0)
    판매구매금액['buy_sell_ratio'] = (판매구매금액.buy_price + 1) /  (판매구매금액.sell_price + 1)

    # 하루 평균 거래량(구매 + 판매)
    ee1 = trade.pivot_table(index = ['day', 'source_acc_id'], values = 'item_price', aggfunc='sum').reset_index()
    ee1.rename(columns = {'source_acc_id': 'acc_id', 'item_price': 'sell_price'}, inplace=True)
    ee2 = trade.pivot_table(index = ['day','target_acc_id'], values = 'item_price', aggfunc='sum').reset_index()
    ee2.rename(columns = {'target_acc_id': 'acc_id', 'item_price': 'buy_price'}, inplace=True)
    거래량 = pd.merge(ee1, ee2, on=('day','acc_id'), how='outer').fillna(0)
    거래량['trade_item_price'] = (거래량.sell_price + 거래량.buy_price) ** 2
    
    # 소속 혈맹의 변수 생성
    pled2 = pled.pivot_table(index = ['day','acc_id'], values = ['play_char_cnt', 'combat_char_cnt', 
                                                                 'pledge_combat_cnt','random_attacker_cnt',
                                                                 'random_defender_cnt', 'same_pledge_cnt', 'temp_cnt', 
                                                                 'etc_cnt','combat_play_time', 'non_combat_play_time'],
                             aggfunc='mean').reset_index()
    pled2['pled_active_group'] = (pled2.combat_char_cnt + 1) / (pled2.play_char_cnt + 1)
    pled2['pled_active_war'] = np.log((pled2.pledge_combat_cnt + 1) * (pled2.temp_cnt + 1)* (pled2.etc_cnt+ 1))
    pled2['pled_active_meet'] = (pled2.same_pledge_cnt + 1) / (pled2.pledge_combat_cnt + 1)
    pled2['pled_aggresive'] = (pled2.random_attacker_cnt + 1) / (pled2.random_defender_cnt + 1)
    pled2['pled_combatlike'] = (pled2.combat_play_time + 1) / (pled2.combat_play_time + 2 + pled2.non_combat_play_time)

    # 넣을 혈맹 변수만 추출
    pled3 = pled2.loc[:, ['day','acc_id', 'play_char_cnt', 'pled_active_group', 'pled_active_war', 'pled_active_meet', 
                          'pled_aggresive', 'pled_combatlike']]
    
    # 중간 merge1
    temp = pd.merge(aa, aa2, on=('day', 'acc_id') , how='left')
    temp = pd.merge(temp, aa3, on=('day', 'acc_id'),  how='left')
    temp = pd.merge(temp, pp2, on=('day', 'acc_id'),  how='left')
    temp = pd.merge(temp, pp1, on=('day', 'acc_id'), how='left')
    temp = pd.merge(temp, 판매구매금액.loc[:, ['day','acc_id', 'buy_sell_ratio']], on=('day', 'acc_id'),  how='left')
    temp = pd.merge(temp, 거래량.loc[:, ['day','acc_id', 'trade_item_price']], on=('day', 'acc_id'),  how='left' )
    temp = pd.merge(temp, pled3, on=('day', 'acc_id'),  how='left')
    temp = temp.fillna(0)
    
    # 평균 결제 금액 별 경험치..
    temp['solo_exp_per_as'] = (temp.solo_exp + 0.01) / (temp.pay_mean + 0.01)
    temp['party_exp_per_as'] = (temp.party_exp  + 0.01) / (temp.pay_mean + 0.01)
    temp['quest_exp_per_as'] = (temp.quest_exp + 0.01) / (temp.pay_mean + 0.01)
    temp['fishing_per_as'] = (temp.fishing + 0.01) / (temp.pay_mean + 0.01)

    # 평균 시간 별 경험치
    temp['solo_exp_per_pt'] = (temp.solo_exp + 0.02) / (temp.playtime + 0.02)
    temp['party_exp_per_pt'] = (temp.party_exp + 0.02) / (temp.playtime + 0.02)
    temp['quest_exp_per_pt'] = (temp.quest_exp + 0.02) / (temp.playtime + 0.02)
    temp['fishing_per_pt'] = (temp.fishing + 0.02) / (temp.playtime + 0.02)

    temp= temp.rename(columns = {'play_char_cnt': 'pled_play_char_cnt'})
    
    # 전투 변수 생성
    combat['user_aggressive'] = (combat.random_attacker_cnt + 1) / (combat.random_defender_cnt + 1)
    combat['user_active_meet'] = (combat.same_pledge_cnt + 1) / (combat.pledge_cnt + 1)
    combat['user_active_war'] = np.log((combat.pledge_cnt + 1) * (combat.temp_cnt + 1)* (combat.etc_cnt+ 1))

    com2 = combat.pivot_table(index = ['day','acc_id' ], values = ['num_opponent', 'user_aggressive',
                                                                   'user_active_meet',   'user_active_war' ], 
                              aggfunc='mean').reset_index()
    com2 = com2.rename(columns = {'num_opponent': 'num_opponent_mean'})
    combat['value'] = 1
    cc2 = combat.pivot_table(index = ['day','acc_id'],aggfunc='sum', columns = 'level', values = 'value').reset_index().fillna(0)

    # 중간merge2
    temp2 = pd.merge(temp, com2, on=('day', 'acc_id') , how='left')
    temp2 = pd.merge(temp2, cc2, on=('day', 'acc_id') , how='left')
    temp2 = temp2.fillna(0)

    # 직업 변수
    class1 = combat.pivot_table(index = ['day','acc_id'], columns='class', values = 'value',
                                aggfunc='sum').fillna(0).reset_index()
    class1= class1.rename(columns ={0 : 'class0',1 : 'class1',2 : 'class2',3 : 'class3',4 : 'class4', 
                                    5 : 'class5',6 : 'class6',7 : 'class7' })

    temp2  = pd.merge(temp2, class1, on=('day', 'acc_id'), how='left')
    
    # 서버 cluster 변수
    from sklearn.cluster import KMeans
    sv_activity = activity[['server', 'playtime', 'npc_kill',
                            'solo_exp', 'party_exp', 'quest_exp', 'rich_monster', 'death', 'revive',
                            'exp_recovery', 'fishing', 'private_shop', 'game_money_change',
                            'enchant_count']].groupby('server').mean()
    sv_combat = combat[['server', 'level', 'pledge_cnt',
                        'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
                        'same_pledge_cnt', 'etc_cnt', 'num_opponent']].groupby('server').mean()
    # class 변수의 경우 범주형 변수이기에 제외
    sv_pledge = pled[['server', 'play_char_cnt',
                      'combat_char_cnt', 'pledge_combat_cnt', 'random_attacker_cnt',
                      'random_defender_cnt', 'same_pledge_cnt', 'temp_cnt', 'etc_cnt',
                      'combat_play_time', 'non_combat_play_time']].groupby('server').mean()
    # pledge_id 변수의 경우 범주형 변수 + 특성을 나타내는 수치가 아니기에 제외
    sv_trade = trade[['type', 'server', 'item_amount', 'item_price']].groupby('server').mean()
    # 'source_acc_id', 'source_char_id', 'target_acc_id', 'target_char_id'의 경우 특성을 나타내는 수치가 아니기에 제외
    # item type의 경우 범주형 변수이기에 제외
    # type의 경우 범주형 변수이나 binary한 값으로, '거래가 교환창으로 이루어진 비율'을 의미할 수 있기에 활용.

    sv_merge = pd.concat([sv_activity, sv_combat, sv_pledge, sv_trade], axis=1, sort=True)

    # server별 mean을 하였는데 null값이라는 의미는 해당 서버에는 feature에 대한 값이 없다는 의미이기에 0을 입력
    sv_merge = sv_merge.fillna(0)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    std_sv_merge = scaler.fit_transform(sv_merge)

    optimal_n_cluster = 5
    kmeans = KMeans(n_clusters=optimal_n_cluster, max_iter=100000, random_state=4)
    clusters = kmeans.fit_predict(std_sv_merge)

    cls_sv_merge = pd.DataFrame(std_sv_merge, columns=sv_merge.columns, index=sv_merge.index)
    cls_sv_merge['K_Means'] = clusters

    cls_sv_merge.reset_index(inplace=True)
    cls_sv_merge_map = cls_sv_merge[['index', 'K_Means']]


    activity['value'] = 1
    cls_sv_merge_map = cls_sv_merge_map.rename(columns = {'index': 'server'})
    combat_temp = pd.merge(activity, cls_sv_merge_map, on='server', how='left')

    combat_server2 = combat_temp.pivot_table(index = ['day','acc_id'], columns='K_Means', 
                                             values = 'value').fillna(0).reset_index()
    combat_server2 = combat_server2.rename(columns = {0: 'server_group0', 1 : 'server_group1', 
                                                      2: 'server_group2', 3:'server_group3', 4: 'server_group4'})

    # 중간merge3
    temp3 = temp2
    temp3 = pd.merge(temp3, combat_server2, on=('day','acc_id'), how='left')
    temp3 = temp3.fillna(0)
    
    # 주차별로 묶기

    features = [ 'enchant_count',
                'exp_recovery',            'fishing',  'game_money_change',
                'npc_kill',          'party_exp',           'playtime',
                'private_shop',          'quest_exp',           'solo_exp',
                'char_id',           'pay_mean',            'pay_sum',
                'pay_count',     'buy_sell_ratio',   'trade_item_price',
                'pled_play_char_cnt',  'pled_active_group',    'pled_active_war',
                'pled_active_meet',     'pled_aggresive',    'pled_combatlike',
                'solo_exp_per_as',   'party_exp_per_as',   'quest_exp_per_as',
                'fishing_per_as',    'solo_exp_per_pt',   'party_exp_per_pt',
                'quest_exp_per_pt',     'fishing_per_pt',  'num_opponent_mean',
                'user_active_meet',    'user_active_war',    'user_aggressive',
                0,                    1,                    2,
                3,                    4,                    5,
                6,                    7,                    8,
                9,                   10,                   11,
                12,                   13,                   14,
                15,                   16,                   17,
                'class0',             'class1',             'class2',
                'class3',             'class4',             'class5',
                'class6',             'class7',      'server_group0',
                'server_group1',      'server_group2',      'server_group3',
                'server_group4']

    temp4 = temp3
    temp4 = temp4.pivot_table(index = 'acc_id', columns = 'day', values  = features)
    temp4.columns.names = ['변수', 'day']

    temp4 = temp4.swaplevel("변수", "day", 1).fillna(0)
    final_acc= temp4

    week1 = (final_acc[1]+final_acc[2]+final_acc[3]+final_acc[4]+final_acc[5]+final_acc[6]+final_acc[7])/7
    week2 = (final_acc[8]+final_acc[9]+final_acc[10]+final_acc[11]+final_acc[12]+final_acc[13]+final_acc[14])/7
    week3 = (final_acc[15]+final_acc[16]+final_acc[17]+final_acc[18]+final_acc[19]+final_acc[20]+final_acc[21])/7
    week4 = (final_acc[22]+final_acc[23]+final_acc[24]+final_acc[25]+final_acc[26]+final_acc[27]+final_acc[28])/7

    # column에 대해 주차 추가
    i=1
    for m in (week1, week2, week3, week4):
        new_col = []
        for k in m.columns:
            l= str(k) +'_week'+ str(i)
            new_col.append(l)
        # print(new_col)
        m.columns = new_col
        i+=1

    # week별로 정보 통합
    result = pd.concat([week1,week2,week3,week4], axis=1).reset_index()
    
    # 한달 요약 정보 추가
    # 마지막 7일에 몇 일 접속했는지
    play_table = activity.pivot_table(index=['acc_id'], aggfunc='count', columns='day', values='playtime')\
        .fillna(0).applymap(lambda x: 1 if x>0 else 0)
    temp = play_table.iloc[:, -7:].sum(axis=1)
    dd2 = pd.DataFrame({'acc_id': temp.index, 'recent_7': temp.values})
    result2 = pd.merge(result, dd2, on='acc_id', how='left')

    # 계정당 캐릭터 수
    char_id = activity.loc[:, ['acc_id', 'char_id']].drop_duplicates().\
    pivot_table(index = 'acc_id', aggfunc='count', values = 'char_id').reset_index()
    result2 = pd.merge(result2, char_id, on='acc_id', how='left')

    # playtime에 대한 세부 변수 추가
    # 한 달의 play 일수
    tt1 = activity.loc[:, ['acc_id', 'day']].drop_duplicates().pivot_table(index = 'acc_id', 
                                                                           values = 'day', aggfunc='count').reset_index()
    tt1 = tt1.rename(columns ={'day': 'pt_day_count'})

    # 한 달의 최대, 최소, 중앙값, 평균, 표준편차 playtime
    tt2 = activity.pivot_table(index = ['acc_id'], aggfunc=['max', 'min', 'median', 'mean', 'std'], 
                               values = 'playtime' ).reset_index()
    tt2.columns = ['acc_id', 'total_max_pt', 'total_min_pt', 'total_median_pt', 'total_mean_pt', 'total_std_pt']
    tt2.fillna(0, inplace=True)

    # merge
    result2 = pd.merge(result2, tt1, on='acc_id', how='left')
    result2 = pd.merge(result2, tt2, on='acc_id', how='left').fillna(0)
    
    # 결제 금액 세부 변수 추가
    # 한 달의 최대, 최소, 중앙값, 평균, 표준편차 결제금액
    pp3 = pay.pivot_table(index = ['acc_id'], aggfunc=['max', 'min', 'median', 'mean', 'std'], 
                          values = 'amount_spent' ).reset_index()
    pp3 = pp3.rename(columns = {'max': 'total_max_pay', 'min': 'total_min_pay', 'median': 'total_median_pay',
                                'mean': 'total_mean_pay', 'std': 'total_std_pay'}).fillna(0)

    # 결제금액 최소/최대 비율
    pp3['total_max_min_ratio_pay'] = pp3.total_min_pay + 0.01 / pp3.total_max_pay +0.01
    pp3.columns = ['acc_id', 'total_max_pay', 'total_min_pay', 'total_median_pay',
                   'total_mean_pay', 'total_std_pay', 'total_max_min_ratio_pay']

    # 총 결제 일수
    pp5 = pay.pivot_table(index = 'acc_id', aggfunc='count', values = 'day').reset_index()
    pp5 = pp5.rename(columns = {'day': 'total_pay_count'})

    result3 = pd.merge(result2, pp3, on='acc_id', how='left')
    result3 = pd.merge(result3, pp5, on='acc_id', how='left')

    # 총 결제 금액 = 계정당 횟수당 평균 결제액 * 결제 횟수
    result3['total_as'] = result3['total_mean_pay'] * result3.total_pay_count

    # 한 캐릭터 당 평균 결제 금액
    result3['total_as_per_char'] = result3.total_as / result3.char_id

    # 총playtime 대비 총 결제 금액
    totalplaytime = activity.pivot_table(index = 'acc_id', aggfunc = 'sum', values = 'playtime').reset_index()
    totalplaytime= totalplaytime.rename(columns = {'playtime': 'total_playtime'})
    result3 = pd.merge(result3, totalplaytime, on='acc_id', how='left')
    result3 = result3.fillna(0)
    result3['total_as_per_total_play_time'] = result3.total_as / result3.total_playtime
    result3 = result3.fillna(0)

    # 총 접속 일수 대비 총 결제 금액
    result3['total_as_per_total_pt'] = result3.total_as / result3.pt_day_count


    # 첫 결제일과 마지막 결제일을 알기 위한 테이블 생성
    spentday_table = pay.pivot_table(index=['acc_id'], aggfunc='count', columns='day', values='amount_spent')\
        .fillna(0).applymap(lambda x: 1 if x>0 else 0)
    # 결제 시작일로부터 지난날
    result3 = result3.set_index('acc_id')
    result3['days_after_firstspent'] = 28 - spentday_table.apply(lambda x: list(x).index(1), axis=1)
    result3 = result3.reset_index()

    # 마지막 결제일로부터 지난날
    result3 = result3.set_index('acc_id')
    result3['days_after_lastspent'] = spentday_table.apply(lambda x: list(x)[::-1].index(1), axis=1) + 1
    result3 = result3.reset_index()

    # days_after_firstspent와 days_after_lastspent의 null값 지우기
    result3.fillna(0, inplace=True)

    # 캐릭터 수 대비 혈맹 가입 수

    pled_id_cnt = pled.loc[:,['acc_id', 'pledge_id'] ].drop_duplicates().\
    pivot_table(index = 'acc_id', values = 'pledge_id', aggfunc='count').reset_index().fillna(0)
    pled_id_cnt = pled_id_cnt.rename(columns = {'pledge_id': 'pled_id_cnt'})

    result3 = pd.merge(result3, pled_id_cnt, on='acc_id', how='left').fillna(0)
    result3['pled_id_cnt'] = result3['pled_id_cnt'] / result3.char_id

    # enchant_count, exp_recovery 세부 변수 추가
    # 각 max, std,mean, median 추가

    enc_exp = activity.pivot_table(index = 'acc_id', values = ['enchant_count', 'exp_recovery'],
                                   aggfunc=['max', 'std', 'mean', 'median']).reset_index().fillna(0)
    enc_exp.columns = ['acc_id', 'total_max_enc', 'total_max_exp_rec', 'total_std_enc', 'total_std_exp_rec',
                       'total_mean_enc', 'total_mean_exp_rec', 'total_median_enc', 'total_median_exp_rec']
    result3 = pd.merge(result3, enc_exp, on='acc_id', how='left')

    # 추가 가중치 주기(서버0, private_shop, recent_7은 feature_importance에서 중요하게 나옴)

    result3.server_group0_week1 = result3.server_group0_week1 ** 2
    result3.server_group0_week2 =result3.server_group0_week2 ** 2
    result3.server_group0_week3 =result3.server_group0_week3 ** 2
    result3.server_group0_week4 =result3.server_group0_week4 ** 2

    result3.private_shop_week1  =result3.private_shop_week1  ** 2
    result3.private_shop_week3  =result3.private_shop_week3  ** 2
    result3.private_shop_week4  =result3.private_shop_week4  ** 2
    result3.private_shop_week2  =result3.private_shop_week2  ** 2

    result3.recent_7 = result3.recent_7  ** 2


    # 경험적으로 enchant_count 와 exp_recovery가 중요한 이슈
    result3.enchant_count_week1 = result3.enchant_count_week1 ** 2
    result3.enchant_count_week2 = result3.enchant_count_week2 ** 2
    result3.enchant_count_week3 = result3.enchant_count_week3 ** 2
    result3.enchant_count_week4 = result3.enchant_count_week4 ** 2
    result3.total_max_enc = result3.total_max_enc ** 2
    result3.total_std_enc = result3.total_std_enc ** 2
    result3.total_mean_enc = result3.total_mean_enc ** 2
    result3.total_median_enc = result3.total_median_enc ** 2

    result3.exp_recovery_week1 = result3.exp_recovery_week1 ** 2
    result3.exp_recovery_week2 = result3.exp_recovery_week2 ** 2
    result3.exp_recovery_week3 = result3.exp_recovery_week3 ** 2
    result3.exp_recovery_week4 = result3.exp_recovery_week4 ** 2
    result3.total_max_exp_rec = result3.total_max_exp_rec ** 2
    result3.total_std_exp_rec = result3.total_std_exp_rec ** 2
    result3.total_mean_exp_rec = result3.total_mean_exp_rec ** 2
    result3.total_median_exp_rec = result3.total_median_exp_rec ** 2

    # 주차에 대한 playtime 에 대한 기울기 변수 추가
    result4 = result3
    result4['playtime_chang_w1_w2'] = (result4['playtime_week2'] - result4['playtime_week1'] ) / (result4['playtime_week1']  + 0.02)
    result4['playtime_chang_w2_w3'] = (result4['playtime_week3'] - result4['playtime_week2'] ) / (result4['playtime_week2'] + 0.02)
    result4['playtime_chang_w3_w4'] = (result4['playtime_week4'] - result4['playtime_week3'] ) / (result4['playtime_week3'] + 0.02)
    result4['playtime_chang_w1_w3'] = (result4['playtime_week3'] - result4['playtime_week1'] ) / (result4['playtime_week1'] + 0.02)
    result4['playtime_chang_w2_w4'] = (result4['playtime_week4'] - result4['playtime_week2'] ) / (result4['playtime_week2'] + 0.02)
    result4['playtime_chang_w1_w4'] = (result4['playtime_week4'] - result4['playtime_week1'] ) / (result4['playtime_week1'] + 0.02)

    # 전체 변수 갖고 Cluster 추가
    # 집단에 대한 cluster 변수 추가
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # clustering data standardization
    cls_data = result4.iloc[:, 1:]
    scaler = StandardScaler()
    std_cls_data = scaler.fit_transform(cls_data)

    # K-Means Clustering
    optimal_n_cluster = 3
    kmeans = KMeans(n_clusters=optimal_n_cluster, max_iter=1000, random_state=4)
    clusters = kmeans.fit_predict(std_cls_data)

    cls_result = result4.copy()
    cls_result['cls'] = clusters

    # get dummies of clusters
    cls_dummy = pd.get_dummies(cls_result.cls, prefix='cls')
    preprocessed_df = pd.concat([result4, cls_dummy], axis=1)

    return preprocessed_df


def amount_spent_preprocessing(activity, pay, trade, pled, combat):
    # 접속일수 관련 column 추가
    play_table = activity.pivot_table(index=['acc_id'], aggfunc='count', columns='day', values='playtime')\
                    .fillna(0).applymap(lambda x: 1 if x>0 else 0)

    # 한 달 내 접속일수
    temp = play_table.sum(axis=1)
    aa = pd.DataFrame({'acc_id': temp.index, 'numofdays': temp.values})

    # 마지막 7일에 몇 일 접속했는지
    temp = play_table.iloc[:, -7:].sum(axis=1)
    aa2 = pd.DataFrame({'acc_id': temp.index, 'recent_7': temp.values})

    # 일일 평균 플레이 시간
    aa3 = activity.pivot_table(index = 'acc_id', values = 'playtime', aggfunc='mean').reset_index()
    aa3 = aa3.rename(columns = {'playtime': 'meanPlaytime'})

    # 각 계정별 하루씩의 평균 플레이 시간(캐릭터 간 차이를 평균으로 봄)
    activity_playtime = activity.pivot_table(index = 'acc_id', columns='day', 
                                             values = 'playtime', aggfunc='mean').fillna(0).reset_index()
    activity_playtime.rename(columns={1 : 'playtime_day1', 2 : 'playtime_day2', 3 : 'playtime_day3', 4 : 'playtime_day4', 5 : 'playtime_day5', 6 : 'playtime_day6', 7 : 'playtime_day7', 8 : 'playtime_day8', 9 : 'playtime_day9', 10 : 'playtime_day10', 11 : 'playtime_day11', 12 : 'playtime_day12', 13 : 'playtime_day13', 14 : 'playtime_day14', 15 : 'playtime_day15', 16 : 'playtime_day16', 17 : 'playtime_day17', 18 : 'playtime_day18', 19 : 'playtime_day19', 20 : 'playtime_day20', 21 : 'playtime_day21', 22 : 'playtime_day22', 23 : 'playtime_day23', 24 : 'playtime_day24', 25 : 'playtime_day25', 26 : 'playtime_day26', 27 : 'playtime_day27', 28 : 'playtime_day28'},
                             inplace=True)

    # 한달 평균 solo_exp
    solo_exp_mean = activity.pivot_table(index = 'acc_id', values=['solo_exp'], aggfunc='mean')
    solo_exp_mean.reset_index(inplace=True)
    solo_exp_mean.rename(columns = {'solo_exp': 'solo_exp_mean'}, inplace=True)

    # 한달 평균 party_exp
    party_exp_mean = activity.pivot_table(index = 'acc_id', values='party_exp', aggfunc='mean')
    party_exp_mean.reset_index(inplace=True)
    party_exp_mean.rename(columns = {'party_exp': 'party_exp_mean'}, inplace=True)

    # 한달 평균 quest_exp
    quest_exp_mean = activity.pivot_table(index = 'acc_id', values='quest_exp', aggfunc='mean')
    quest_exp_mean.reset_index(inplace=True)
    quest_exp_mean.rename(columns={'quest_exp': 'quest_exp_mean'}, inplace=True)

    # 한달 평균 fishing_time
    fishing_mean = activity.pivot_table(index = 'acc_id', values='fishing', aggfunc='mean')
    fishing_mean.reset_index(inplace=True)
    fishing_mean.rename(columns={'fishing': 'fishing_mean'}, inplace=True)

    # 한달 평균 private_shop
    private_shop_mean = activity.pivot_table(index = 'acc_id', values='private_shop', aggfunc='mean')
    private_shop_mean.reset_index(inplace=True)
    private_shop_mean.rename(columns={'private_shop': 'private_shop_mean'}, inplace=True)

    # 한달 평균 exp_recovery
    enc_exp = activity.pivot_table(index = 'acc_id', values = 'exp_recovery',
                                   aggfunc= 'mean').reset_index().fillna(0)
    enc_exp.rename(columns={'exp_recovery': 'exp_recovery_mean'}, inplace=True)

    # 캐릭터 갯수
    char_count = activity.pivot_table(index = 'acc_id', values = 'char_id', aggfunc = 'count').reset_index()
    char_count = char_count.fillna(0)

    # 중간 merge1
    temp = pd.merge(aa, aa2 , on='acc_id')
    temp = pd.merge(temp, aa3, on='acc_id')
    temp = pd.merge(temp,activity_playtime, on='acc_id')
    temp = pd.merge(temp, solo_exp_mean, on='acc_id')
    temp = pd.merge(temp, party_exp_mean, on='acc_id')
    temp = pd.merge(temp, quest_exp_mean, on='acc_id')
    temp = pd.merge(temp, fishing_mean, on='acc_id')
    temp = pd.merge(temp, private_shop_mean, on='acc_id')
    temp = pd.merge(temp, enc_exp, on='acc_id', how='left').fillna(0)
    temp = pd.merge(temp, char_count, on='acc_id')
    temp = temp.fillna(0)

    # 한달 평균 결제금액
    pay2 = pay.pivot_table(index = 'acc_id', aggfunc='mean', values = 'amount_spent' ).reset_index()
    pay2 = pay2.rename(columns = {'amount_spent': 'pay_mean'})
    # 중간 merge2
    temp = pd.merge(temp, pay2, on='acc_id', how='left').fillna(0)

    # 평균 결제 금액 별 경험치..
    temp['solo_exp_per_as'] = (temp.solo_exp_mean + 0.01) / (temp.pay_mean + 0.01)
    temp['party_exp_per_as'] = (temp.party_exp_mean + 0.01) / (temp.pay_mean + 0.01)
    temp['quest_exp_per_as'] = (temp.quest_exp_mean + 0.01) / (temp.pay_mean + 0.01)

    # 거래 테이블 한달 평균 sell_price
    aa = trade.pivot_table(index = 'source_acc_id', values = 'item_price', aggfunc='mean').reset_index()
    aa.rename(columns = {'source_acc_id': 'acc_id', 'item_price': 'sell_price'}, inplace=True)

    # 거래 테이블 한달 평균 buy_price
    aa2 = trade.pivot_table(index = 'target_acc_id', values = 'item_price', aggfunc='mean').reset_index()
    aa2.rename(columns = {'target_acc_id': 'acc_id', 'item_price': 'buy_price'}, inplace=True)

    # sell_price + buy_price merge
    price = pd.merge(aa, aa2, on='acc_id', how='outer').fillna(0)

    # 판매 대비 구매한 비율 변수 생성
    price['buy_sell_ratio'] = (price.buy_price + 1) /  (price.sell_price + 1)

    # 중간 merge3
    temp = pd.merge(temp, price.loc[:, ['acc_id', 'buy_sell_ratio']], on='acc_id', how='left').fillna(0)

    # 소속 혈맹의 변수 생성
    pled2 = pled.pivot_table(index = 'acc_id', values = ['play_char_cnt', 'combat_char_cnt', 
                                                         'pledge_combat_cnt','random_attacker_cnt',
                                                         'random_defender_cnt', 'same_pledge_cnt', 'temp_cnt', 
                                                         'etc_cnt','combat_play_time', 'non_combat_play_time' ],
                             aggfunc='mean').reset_index()

    pled2['pled_active_group'] = (pled2.combat_char_cnt + 1) / (pled2.play_char_cnt + 1)
    pled2['pled_active_war'] = np.log((pled2.pledge_combat_cnt + 1) * (pled2.temp_cnt + 1)* (pled2.etc_cnt+ 1))
    pled2['pled_active_meet'] = (pled2.same_pledge_cnt + 1) / (pled2.pledge_combat_cnt + 1)
    pled2['pled_aggresive'] = (pled2.random_attacker_cnt + 1) / (pled2.random_defender_cnt + 1)
    pled2['pled_combatlike'] = (pled2.combat_play_time + 1) / (pled2.combat_play_time + 2 + pled2.non_combat_play_time)

    # 넣을 혈맹 변수만 추출
    pled3 = pled2.loc[:, ['acc_id', 'play_char_cnt', 'pled_active_group', 'pled_active_war', 
                          'pled_active_meet', 'pled_aggresive', 'pled_combatlike']]

    # 중간 merge4
    temp2 = pd.merge(temp, pled3, on='acc_id', how='left').fillna(0)

    # 전투 변수 생성
    combat['user_aggressive'] = (combat.random_attacker_cnt + 1)  /  (combat.random_defender_cnt + 1)
    combat['user_active_meet'] = (combat.same_pledge_cnt + 1) /(combat.pledge_cnt + 1)
    combat['user_active_war'] = np.log((combat.pledge_cnt + 1) * (combat.temp_cnt + 1)* (combat.etc_cnt+ 1))

    # 전투에서 한달 평균 num_opponent
    com2 = combat.pivot_table(index = 'acc_id', values = 'num_opponent', aggfunc='mean').reset_index()
    com2 = com2.rename(columns = {'num_opponent': 'num_opponent_mean'})

    # 계정 별 한 char_id의 최소 level, 최대 level
    cc2 = combat.pivot_table(index = 'acc_id', aggfunc=['min','max'], values = 'level').reset_index()
    cc2.columns = ['acc_id', 'min_level', 'max_level']
    for_max = combat.pivot_table(index = ['acc_id', 'char_id'], aggfunc=['max'], values = 'level')
    for_min = combat.pivot_table(index = ['char_id'], aggfunc=['min'], values = 'level')
    for_max.columns = ['max_level']
    for_min.columns = ['min_level']
    cc4 = combat[['acc_id', 'char_id', 'level']]

    temp = pd.merge(cc4, cc2[['acc_id', 'max_level']], on='acc_id', how='left')
    temp['check'] = temp['level'] == temp['max_level']
    temp = temp[temp['check']].drop_duplicates(subset=['acc_id'])

    level = for_max.loc[temp.set_index(['acc_id', 'char_id']).index].reset_index()\
        .drop('char_id', axis=1)
    level['min_level'] = for_min.loc[temp.set_index(['char_id']).index]['min_level'].values
    level = level[['acc_id', 'min_level', 'max_level']]

    # 직업 변수
    combat['value'] = 1
    class1 = combat.pivot_table(index = 'acc_id', columns='class', values = 'value', aggfunc='sum').fillna(0).reset_index()
    class1= class1.rename(columns ={0 : 'class0',1 : 'class1',2 : 'class2',3 : 'class3', 4 : 'class4',5 : 'class5', 
                                    6 : 'class6',7 : 'class7'})

    # combat테이블에서 생성한 변수들 중간merge 5
    combat3 = combat.loc[: ,['acc_id', 'user_aggressive', 'user_active_meet','user_active_war']].\
        pivot_table(index = 'acc_id', aggfunc='mean', values = [ 'user_aggressive', \
                                                                'user_active_meet','user_active_war']).reset_index()

    com2 = combat.pivot_table(index = 'acc_id', values = 'num_opponent', aggfunc='mean').reset_index()
    com2 = com2.rename(columns = {'num_opponent': 'num_opponent_mean'})

    temp3 = pd.merge(temp2, combat3, on='acc_id', how='left').fillna(0)
    temp3 = pd.merge(temp3, com2, on='acc_id', how='left').fillna(0)
    temp3 = pd.merge(temp3, level, on='acc_id', how='left').fillna(0)
    temp3 = pd.merge(temp3, class1, on='acc_id', how='left').fillna(0)

    # 평균결제 금액 대비 level 차이
    temp3['level_change_pay_weighted'] = (temp3.max_level - temp3.min_level + 0.01) / (temp3.pay_mean + 0.01)

    # 평균playtime 대비 level 차이
    temp3['level_change_time_weighted'] = (temp3.max_level - temp3.min_level + 0.02) / (temp3.meanPlaytime + 0.02)

    ## 서버 cluster 변수
    from sklearn.cluster import KMeans
    sv_activity = activity[['server', 'playtime', 'npc_kill',
                            'solo_exp', 'party_exp', 'quest_exp', 'rich_monster', 'death', 'revive',
                            'exp_recovery', 'fishing', 'private_shop', 'game_money_change',
                            'enchant_count']].groupby('server').mean()
    sv_combat = combat[['server', 'level', 'pledge_cnt',
                        'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
                        'same_pledge_cnt', 'etc_cnt', 'num_opponent']].groupby('server').mean()
    # class 변수의 경우 범주형 변수이기에 제외
    sv_pledge = pled[['server', 'play_char_cnt',
                      'combat_char_cnt', 'pledge_combat_cnt', 'random_attacker_cnt',
                      'random_defender_cnt', 'same_pledge_cnt', 'temp_cnt', 'etc_cnt',
                      'combat_play_time', 'non_combat_play_time']].groupby('server').mean()
    # pledge_id 변수의 경우 범주형 변수 + 특성을 나타내는 수치가 아니기에 제외
    sv_trade = trade[['type', 'server', 'item_amount', 'item_price']].groupby('server').mean()
    # 'source_acc_id', 'source_char_id', 'target_acc_id', 'target_char_id'의 경우 특성을 나타내는 수치가 아니기에 제외
    # item type의 경우 범주형 변수이기에 제외
    # type의 경우 범주형 변수이나 binary한 값으로, '거래가 교환창으로 이루어진 비율'을 의미할 수 있기에 활용.

    sv_merge = pd.concat([sv_activity, sv_combat, sv_pledge, sv_trade], axis=1, sort=True)

    # server별 mean을 하였는데 null값이라는 의미는 해당 서버에는 feature에 대한 값이 없다는 의미이기에 0을 입력
    sv_merge = sv_merge.fillna(0)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    std_sv_merge = scaler.fit_transform(sv_merge)

    optimal_n_cluster = 5
    kmeans = KMeans(n_clusters=optimal_n_cluster, max_iter=100000, random_state=4)
    clusters = kmeans.fit_predict(std_sv_merge)

    cls_sv_merge = pd.DataFrame(std_sv_merge, columns=sv_merge.columns, index=sv_merge.index)
    cls_sv_merge['K_Means'] = clusters

    cls_sv_merge.reset_index(inplace=True)
    cls_sv_merge_map = cls_sv_merge[['index', 'K_Means']]

    cls_sv_merge_map = cls_sv_merge_map.rename(columns = {'index': 'server'})
    combat_temp = pd.merge(activity, cls_sv_merge_map, on='server', how='left')
    combat_temp['value'] = 1
    combat_server2 = combat_temp.pivot_table(index = 'acc_id', columns='K_Means' , values = 'value').fillna(0).reset_index()

    # 중간 merge6
    temp4 = temp3
    temp4 = pd.merge(temp4, combat_server2, on='acc_id', how='left')
    temp4 = temp4.rename(columns = {0: 'server_group0', 1 : 'server_group1', 2: 'server_group2', 
                                    4: 'server_group4', 3: 'server_group3'})

    # 한달 평균 결제 횟수
    pp5 = pay.pivot_table(index = 'acc_id', aggfunc='count', values = 'day').reset_index()
    pp5 = pp5.rename(columns = {'day': 'total_pay_count'})
    temp4 = pd.merge(temp4, pp5, on='acc_id', how='left')

    # 시간대비 fishint_time , 시간대비 private_shop
    temp4['fishing_per_pt'] = (temp4.fishing_mean + 0.02) / (temp4.meanPlaytime + 0.02)
    temp4['private_shop_per_pt'] = (temp4.private_shop_mean + 0.02) / (temp4.meanPlaytime + 0.02)

    # 한달 총 결제금액
    temp4['total_as'] = temp4.pay_mean * temp4.total_pay_count

    preprocessed_df = temp4.fillna(0)

    return preprocessed_df


def train_preprocessing(label, activity, pay, trade, pled, combat):
    st_df = survival_time_preprocessing(activity, pay, trade, pled, combat)
    as_df = amount_spent_preprocessing(activity, pay, trade, pled, combat)

    # 학습을 위해 label을 붙이기
    st_df = pd.merge(st_df, label, on='acc_id', how='left')
    as_df = pd.merge(as_df, label, on='acc_id', how='left')

    return st_df, as_df

def test_preprocessing(activity, pay, trade, pled, combat):
    st_df = survival_time_preprocessing(activity, pay, trade, pled, combat)
    as_df = amount_spent_preprocessing(activity, pay, trade, pled, combat)
    return st_df, as_df

def save_dataframe(df, f_name):
    df.to_csv(f_name, encoding='utf-8', index=False)

if __name__ == '__main__':
    # 데이터 load
    dataset = DataSet()

    # train/test 데이터 전처리
    train_survival, train_spent = train_preprocessing(*dataset.get_train_data())
    test1_survival, test1_spent = test_preprocessing(*dataset.get_test1_data())
    test2_survival, test2_spent = test_preprocessing(*dataset.get_test2_data())
 
    # 데이터 저장
    save_dataframe(train_survival, "train_preprocess_1.csv")
    save_dataframe(train_spent, "train_preprocess_2.csv")
    save_dataframe(test1_survival, "test1_preprocess_1.csv")
    save_dataframe(test1_spent, "test1_preprocess_2.csv")
    save_dataframe(test2_survival, "test2_preprocess_1.csv")
    save_dataframe(test2_spent, "test2_preprocess_2.csv")