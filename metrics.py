"""
Master metrics for measuring ML model performance
"""

import numpy as np

class ClassificationMetric():

    def accuracy_score(y_true, y_pred):
        
        """Accuracy"""

        return np.mean(y_true == y_pred)

    def precision_score(y_true, y_pred):
        
        """Precision"""
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall_score(y_true, y_pred):
        
        """Recall"""
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def confusion_matrix(y_true, y_pred):
        
        """Confusion Matrix"""
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[tn, fp],
                         [fn, tp]])
    
    def f1_score(self, y_true, y_pred):

        """F1 score"""

        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def roc_auc_score(y_true, y_scores):

        """ROC AUC using the trapezoidal rule"""

        # Sort scores and corresponding true labels
        desc_score_indices = np.argsort(-y_scores)
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]

        tpr = []
        fpr = []

        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)

        tp = 0
        fp = 0
        prev_score = -np.inf

        for i in range(len(y_true)):
            if y_scores[i] != prev_score:
                tpr.append(tp / P if P else 0)
                fpr.append(fp / N if N else 0)
                prev_score = y_scores[i]
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
        tpr.append(tp / P if P else 0)
        fpr.append(fp / N if N else 0)

        # Compute area under curve
        return np.trapz(tpr, fpr)
    
    def log_loss(y_true, y_probs, eps=1e-15):

        """Log Loss / Cross Entropy"""

        y_probs = np.clip(y_probs, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_probs) + (1 - y_true) * np.log(1 - y_probs))
    

class RegressionMetric():

    def mean_absolute_error(y_true, y_pred):

        """Mean Absolute Error"""

        return np.mean(np.abs(y_true - y_pred))
    
    def mean_squared_error(y_true, y_pred):

        """Mean Squared Error"""

        return np.mean((y_true - y_pred) ** 2)
    
    def root_mean_squared_error(self, y_true, y_pred):

        """Root Mean Squared Error"""

        return np.sqrt(self.mean_squared_error(y_true, y_pred))

    def mean_absolute_percentage_error(y_true, y_pred):

        """Mean Absolute Prct Error (MAPE)"""

        y_true = np.where(y_true == 0, 1e-10, y_true)  # avoid division by zero
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def r2_score(y_true, y_pred):

        """R2 Square"""

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total)

    def adjusted_r2_score(self, y_true, y_pred, n_features):

        """R2 Square (Adjusted)"""

        n = len(y_true)
        r2 = self.r2_score(y_true, y_pred)
        return 1 - ((1 - r2) * (n - 1)) / (n - n_features - 1)

