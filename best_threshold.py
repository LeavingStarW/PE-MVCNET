import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

pickle_path = 'data/pickle/preds.pickle'

file = open(pickle_path,'rb')
data = pickle.load(file)


'''caculate best threshold'''
def cal(data,threshold,will_print=False):
    pred_list = []
    label_list = []
    for i in data.values():
        pred_list.append(i['pred'])
        label_list.append(i['label'])

    threshold=threshold
    sum_correct=0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in data.values():
        flag=0
        if i['pred']>threshold:
            flag=1
        if i['label']==flag:
            sum_correct+=1

        # TP
        if i['pred']>threshold and i['label']==1:
            TP+=1
        # TN
        elif i['pred']<=threshold and i['label']==0:
            TN+=1
        # FP
        elif i['pred']>threshold and i['label']==0:
            FP+=1
        # FN
        elif i['pred']<=threshold and i['label']==1:
            FN+=1

    acc = {'indicator':'acc','data':sum_correct / len(data)}
    specificity = {'indicator':'specificity','data':TN / (TN + FP + 1e-100)}
    sensitivity = {'indicator':'sensitivity','data':TP / (TP + FN + 1e-100)}
    PPV = {'indicator':'PPV','data':TP / (TP + FP + 1e-100)}
    NPV = {'indicator':'NPV','data':TN / (TN + FN + 1e-100)}
    f1 = {'indicator':'f1','data':(2 * PPV['data'] * sensitivity['data']) / (PPV['data'] + sensitivity['data'] + 1e-100)}

    # choose the indicator
    indicator = acc
    if will_print:
        print('Indicator:\t',indicator['indicator'])
        print('AUROC\t{:.3f}'.format(roc_auc_score(label_list, pred_list)))
        print('ACC\t{:.3f}'.format(acc['data']))
        print('F1\t{:.3f}'.format(f1['data']))
        print('Sensitivity\t{:.3f}'.format(sensitivity['data']))
        print('Specificity\t{:.3f}'.format(specificity['data']))
        print('NPV\t{:.3f}'.format(NPV['data']))
        print('PPV\t{:.3f}'.format(PPV['data']))
    else:
        return indicator


'''调用'''
best_threshold = 0
best_indicator = 0
for i in np.arange(0, 1, 0.001):
    indicator = cal(data, i)
    if indicator['data'] > best_indicator:
        best_indicator = indicator['data']
        best_threshold = i


# main
print('best_threshold:\t', best_threshold)
cal(data, best_threshold, True)
