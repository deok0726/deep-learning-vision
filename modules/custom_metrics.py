from sklearn import metrics

class Metrics(object):
    def __init__(self, target_label, unique_anomaly):
        self.target_label = target_label
        self.unique_anomaly = unique_anomaly

    def __call__(self, test_label):
        try:
            if(self.unique_anomaly):
                test_label[test_label == self.target_label] = -1
                test_label[test_label != -1] = 1
            else:
                test_label[test_label != self.target_label] = -1
                test_label[test_label != -1] = 1
        except Exception as e:
            print(e)

class AUROC(Metrics):
    def __init__(self, target_label, unique_anomaly):
        super().__init__(target_label, unique_anomaly)

    def __call__(self, score, test_label):
        super().__call__(test_label)
        try:
            fprs, tprs, _ = metrics.roc_curve(test_label, score, -1)
            return metrics.auc(fprs, tprs)
        except Exception as e:
            print(e)

class F1(Metrics):
    def __init__(self, target_label, unique_anomaly):
        super().__init__(target_label, unique_anomaly)

    def __call__(self, score, test_label, threshold):
        super().__call__(test_label)
        try:
            pred = score > threshold
            pred = pred.astype(int)
            pred[pred==1] = -1
            pred[pred==0] = 1
            f1_score = metrics.f1_score(test_label, pred, -1)
            return f1_score
        except Exception as e:
            print(e)