from sklearn import metrics
from torchvision.transforms.functional import normalize

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
            return test_label
        except Exception as e:
            print(e)

class AUROC(Metrics):
    def __init__(self, target_label, unique_anomaly):
        super().__init__(target_label, unique_anomaly)

    def __call__(self, score, test_label):
        test_label = super().__call__(test_label)
        score = ((score - score.min(axis=0))/(score.max(axis=0) - score.min(axis=0)))
        try:
            fprs, tprs, _ = metrics.roc_curve(test_label, score, pos_label=-1)
            return metrics.auc(fprs, tprs)
        except Exception as e:
            print(e)

class AUPRC(Metrics):
    def __init__(self, target_label, unique_anomaly):
        super().__init__(target_label, unique_anomaly)

    def __call__(self, score, test_label):
        test_label = super().__call__(test_label)
        score = ((score - score.min(axis=0))/(score.max(axis=0) - score.min(axis=0)))
        try:
            precision, recall, _ = metrics.precision_recall_curve(test_label, score, pos_label=-1)
            return metrics.auc(recall, precision)
        except Exception as e:
            print(e)

class F1(Metrics):
    def __init__(self, target_label, unique_anomaly):
        super().__init__(target_label, unique_anomaly)

    def __call__(self, score, test_label, threshold):
        test_label = super().__call__(test_label)
        try:
            pred = score > threshold
            pred = pred.astype(int)
            pred[pred==1] = -1
            pred[pred==0] = 1
            f1_score = metrics.f1_score(test_label, pred, pos_label=-1)
            return f1_score
        except Exception as e:
            print(e)

def classification_report(score, test_label, threshold, target_label=0, unique_anomaly=False):
    try:
        if(unique_anomaly):
            test_label[test_label == target_label] = -1
            test_label[test_label != -1] = 1
        else:
            test_label[test_label != target_label] = -1
            test_label[test_label != -1] = 1
        pred = score > threshold
        pred = pred.astype(int)
        pred[pred==1] = -1
        pred[pred==0] = 1
        print(metrics.confusion_matrix(test_label, pred, [-1, 1]))
        return metrics.classification_report(test_label, pred, [-1, 1], ['Anomaly', 'Normal'])
    except Exception as e:
        print(e)