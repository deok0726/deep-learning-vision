from sklearn import metrics

class ROC(object):
    def __init__(self, target_label):
        self.target_label = target_label
        print('roc object init')
    def __call__(self, score, test_label):
        try:
            test_label[test_label != self.target_label] = 0
            test_label[test_label == self.target_label] = 1
            fprs, tprs, _ = metrics.roc_curve(test_label, score)
            return metrics.auc(fprs, tprs)
        except Exception as e:
            print(e)
