from sklearn import metrics

class ROC(object):
    def __init__(self, target_label):
        self.target_label = target_label
        print('roc object init')
    def __call__(self, score, test_label):
        try:
            fprs, tprs, _ = metrics.roc_curve(test_label, score, self.target_label)
            return metrics.auc(fprs, tprs)
        except Exception as e:
            print(e)
