from sklearn import metrics

class ROC(object):
    def __init__(self, target_label, unique_anomaly):
        self.target_label = target_label
        self.unique_anomaly = unique_anomaly

    def __call__(self, score, test_label):
        try:
            if(self.unique_anomaly):
                test_label[test_label == self.target_label] = -1
                test_label[test_label != -1] = 1
            else:
                test_label[test_label != self.target_label] = -1
                test_label[test_label != -1] = 1
            fprs, tprs, _ = metrics.roc_curve(test_label, score, -1)
            return metrics.auc(fprs, tprs)
        except Exception as e:
            print(e)
