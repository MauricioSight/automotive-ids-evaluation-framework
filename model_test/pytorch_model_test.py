import os
import torch
import random
import typing
import datetime

import pandas as pd
import numpy as np

from torch import nn

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix as confusion_matrix_sci
import matplotlib.pyplot as plt
import seaborn as sns

from . import abstract_model_test

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryROC,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassROC
)

from custom_metrics import (
    timing,
    storage
)

class PytorchModelTest(abstract_model_test.AbstractModelTest):
    def __init__(self, model, model_specs_dict: typing.Dict):
        self._model = model
        self._labeling_schema = model_specs_dict['feat_gen']['config']['labeling_schema']
        self._model_name = model_specs_dict['model_specs']['model_name']
        self._model_type = model_specs_dict['model_specs']['model_type']
        self._presaved_models_state_dict = model_specs_dict['model_specs']['presaved_paths']
        self._model_specs_dict = model_specs_dict['model_specs']
        self._evaluation_metrics = []
        self._batch_size = model_specs_dict['model_specs']['hyperparameters']['batch_size']
        self._threshold = model_specs_dict['model_specs']['hyperparameters']['threshold']
        self._number_of_outputs = model_specs_dict['model_specs']['hyperparameters'].get('num_outputs', -1)
        # self._forward_output_path = model_specs_dict['model_specs']['paths']['forward_output_path']
        self._confusion_matrix = None
        self._roc_metrics = None

        self._criterion = model_specs_dict['model_specs']['criterion']

        self._window_size = model_specs_dict['feat_gen']['config']['window_size']
        self._number_of_bytes = model_specs_dict['feat_gen']['config']['number_of_bytes']
        self._feature_generator = model_specs_dict['feat_gen']['feature_generator']
        self._multiclass = model_specs_dict['feat_gen']['config']['multiclass']
        self._dataset = model_specs_dict['feat_gen']['config']['dataset']

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_pytorch_test"

        # TODO: Get this from json config file
        art_path = "./workspace/artifacts"
        self._artifacts_path = f"{art_path}/{self._run_id}"

        if not os.path.exists(self._artifacts_path):
            os.makedirs(self._artifacts_path)
            print("Artifacts output directory created successfully")

        self._metrics_output_path = f"{self._artifacts_path}/metrics"
        if not os.path.exists(self._metrics_output_path):
            os.makedirs(self._metrics_output_path)
            print("Metrics output directory created successfully")


    def __seed_all(self, seed):
        # Reference
        # https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
        if not seed:
            seed = 10

        print("[ Using Seed : ", seed, " ]")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def __seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def __model_cnn_forward(self, device, testloader, fold):
        self._model.eval()

        print(">> Executing forward to export")

        N_OF_ENTRIES = len(testloader) * self._batch_size
        # Number of input features for the flatten layer

        # store_tensor = torch.zeros((N_OF_ENTRIES, N_OF_FEATURES), dtype=torch.float32)
        print(f"store_tensor.shape = {store_tensor.shape}")

        print(">> Preallocated output tensor")
        store_tensor_index = 0

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = self._model.fc1_forward(data)
                output = output.detach()

                start_index = store_tensor_index
                end_index = store_tensor_index + self._batch_size
                for index in range(0, self._batch_size):
                    # isso aqui pode ta copiando a referencia
                    # e mantendo os valores sempres iguais
                    store_tensor[start_index + index] = output[index].clone()

                store_tensor_index = store_tensor_index + self._batch_size

        store_tensor = store_tensor.cpu().numpy()

        np.savez(f"{self._forward_output_path}/sample_model_fold_{fold}_fc1_forward.npz", store_tensor)


    def _get_metrics_per_attack(self, fpr_t, fold, threshold):
        total_per_label = fpr_t['y_true'].value_counts().to_dict()
        metrics_per_attack = {}

        for attack_label, total in total_per_label.items():
            if attack_label == 0:
                # Extract true labels and predicted probabilities for benign
                y_t = fpr_t[fpr_t['y_true'] == 0]['y_true']
                y_p = fpr_t[fpr_t['y_true'] == 0]['y_pred']

                print(f'Shape of attack label {attack_label}: {y_t.shape}')

                # Calculate metrics for benign detection
                # aucroc = roc_auc_score(y_t, y_p)
                y_pred_thresholded = (y_p >= threshold).astype(int)

                accuracy = accuracy_score(y_t, y_pred_thresholded)
                precision = precision_score(y_t, y_pred_thresholded)
                recall = recall_score(y_t, y_pred_thresholded)
                f1 = f1_score(y_t, y_pred_thresholded)

                metrics_per_attack[attack_label] = {
                    'AUCROC': 0.,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Threshold': threshold
                }

                continue

            # Extract true labels and predicted probabilities for binary classification (benign vs specific attack)
            y_t = fpr_t[(fpr_t['y_true'] == 0) | (fpr_t['y_true'] == attack_label)]['y_true']
            y_p = fpr_t[(fpr_t['y_true'] == 0) | (fpr_t['y_true'] == attack_label)]['y_pred']

            y_t = y_t.replace({2: 1, 3: 1, 4: 1, 5: 1})

            print(f'Shape of attack label {attack_label}: {y_t.shape}')

            # Calculate AUC-ROC
            aucroc = roc_auc_score(y_t, y_p)

            # Apply the optimal threshold to convert probabilities to binary predictions
            y_pred_thresholded = (y_p >= threshold).astype(int)

            # Calculate other metrics
            accuracy = accuracy_score(y_t, y_pred_thresholded)
            precision = precision_score(y_t, y_pred_thresholded)
            recall = recall_score(y_t, y_pred_thresholded)
            f1 = f1_score(y_t, y_pred_thresholded)

            metrics_per_attack[attack_label] = {
                'AUCROC': aucroc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Threshold': threshold
            }

        # Convert results to DataFrame for easier export and analysis
        aux_df = pd.DataFrame.from_dict(metrics_per_attack, orient='index').reset_index()
        aux_df.columns = ['Class', 'AUCROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Threshold']

        # Map attack labels to human-readable class names
        if self._dataset == 'TOW_IDS_dataset':
            aux_df["Class"] = aux_df["Class"].map(
                {
                    0: "benign",
                    1: 'CAN DoS',
                    2: 'CAN replay',
                    3: 'Switch (MAC Flooding)',
                    4: 'Frame injection',
                    5: 'PTP sync',
                }
            )
        
        elif self._dataset == 'IN_VEHICLE_dataset':
            aux_df["Class"] = aux_df["Class"].map(
                {
                    0: "benign",
                    # 1: 'Flooding',
                    # 2: 'Fuzzy',
                    1: 'Malfunction',
                    # 4: 'Frame injection',
                    2: 'Replay',
                }
            )

        # Save the results to CSV
        aux_df.to_csv(f"{self._metrics_output_path}/metrics_per_attack_{self._labeling_schema}_fold_{fold}_{self._model_name}.csv", index=False)
    
    def _get_threshold_youden_index(self, y_true, y_pred):
        # If tensors are on GPU, move them to CPU and convert to NumPy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        # Calculate Youden index to determine optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_index]
        return optimal_threshold

    def __test_model(self, device, testloader, fold, criterion_without_reduction):
        self._model.eval()

        if self._number_of_outputs > 1:
            accuracy_metric = MulticlassAccuracy(num_classes=self._number_of_outputs).to(device)
            f1_score_metric = MulticlassF1Score(num_classes=self._number_of_outputs).to(device)
            auc_roc_metric = MulticlassAUROC(num_classes=self._number_of_outputs).to(device)
            precision_score = MulticlassPrecision(num_classes=self._number_of_outputs).to(device)
            recall_score = MulticlassRecall(num_classes=self._number_of_outputs).to(device)
            confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=self._number_of_outputs).to(device)
            roc_metric = MulticlassROC(num_classes=self._number_of_outputs, thresholds=1000).to(device)
        else:
            accuracy_metric = BinaryAccuracy().to(device)
            f1_score_metric = BinaryF1Score().to(device)
            auc_roc_metric = BinaryAUROC().to(device)
            precision_score = BinaryPrecision().to(device)
            recall_score = BinaryRecall().to(device)
            confusion_matrix_metric = BinaryConfusionMatrix().to(device)
            roc_metric = BinaryROC(thresholds=1000).to(device)

        # TODO: alterar para prealocar y_pred e y_true pra exeutar mais rapido
        # n_entries = self._batch_size * len(testloader)
        # y_pred = torch.zeros((n_entries, self._number_of_outputs)).to(device)
        # y_true = torch.zeros((n_entries, self._number_of_outputs)).to(device)
        y_pred = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        initial_entry = 0

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = self._model(data)
                output = output.detach()

                if self._model_type in ['anomaly_detection']:
                    loss_elements = criterion_without_reduction(output, data)
                    mean_loss = loss_elements.mean(dim=(1, 2, 3))
                    
                    if self._multiclass:
                        target = torch.argmax(target, dim=1)
                    else:
                        target = target.view(-1)

                    y_pred = torch.cat((y_pred, mean_loss))
                    y_true = torch.cat((y_true, target))

                elif self._model_type == 'classification':
                    y_pred = torch.cat((y_pred, output))
                    y_true = torch.cat((y_true, target))

                    # TODO: Find a better way to perform this computation
                    if self._number_of_outputs == 6:
                        auc_roc_metric.update(output, torch.argmax(target, dim=1))
                    else:
                        auc_roc_metric.update(output, target)

                    # for index in range(self._batch_size):
                    #     y_pred[initial_entry + index] = output[index].clone()
                    #     y_true[initial_entry + index] = target[index].clone()
                    # initial_entry = initial_entry + self._batch_size

                    accuracy_metric.update(output, target)
                    f1_score_metric.update(output, target)
                    precision_score.update(output, target)
                    recall_score.update(output, target)


            # Create a DataFrame from the ROC metrics
            roc_df = pd.DataFrame({
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
            })

            # Save the DataFrame to a CSV file
            roc_df.to_csv(f"{self._metrics_output_path}/y_true__y_pred_{self._labeling_schema}_fold_{fold}_{self._model_name}.csv", index=False)

            # TODO: think a better way to do this
            if self._multiclass and self._model_type in ['anomaly_detection']:
                self._get_metrics_per_attack(roc_df, fold, self._threshold)

                y_true = torch.where((y_true == 2) | (y_true == 3) | (y_true == 4) | (y_true == 5), torch.tensor(1), y_true)

            if self._model_type in ['anomaly_detection']:
                output = y_pred > self._threshold

                auc_roc_metric.update(y_pred, y_true)
                accuracy_metric.update(output, y_true)
                f1_score_metric.update(output, y_true)
                precision_score.update(output, y_true)
                recall_score.update(output, y_true)

                cm = confusion_matrix_sci(y_true.tolist(), output.tolist())
                group_counts = [f'{value:.0f}' for value in confusion_matrix_sci(y_true.tolist(), output.tolist()).ravel()]
                group_percentages = [f'{value*100:.2f}%' for value in confusion_matrix_sci(y_true.tolist(), output.tolist()).ravel()/np.sum(cm)]
                labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
                labels = np.array(labels).reshape(2,2)
                sns.heatmap(cm, annot=labels, cmap='Oranges', xticklabels=['Predicted Benign', 'Predicted Malicious'], yticklabels=['Actual Benign', 'Actual Malicious'], fmt='')
                plt.savefig(f"{self._metrics_output_path}/confusion_matrix_{self._labeling_schema}_fold_{fold}_{self._model_name}.png", dpi=300, bbox_inches='tight')
                plt.close()

            # Calculate metrics
            acc = accuracy_metric.compute().cpu().numpy()
            f1 = f1_score_metric.compute().cpu().numpy()
            roc_auc = auc_roc_metric.compute().cpu().numpy()
            prec = precision_score.compute().cpu().numpy()
            recall = recall_score.compute().cpu().numpy()

            # Reshape y_pred and y_true to compute confusion matrix
            # TODO: Esse reshape deve ocorrer apenas se for necessário
            y_pred_conf_matrix = y_pred
            y_true_conf_matrix = y_true
            if self._number_of_outputs == 6:
                y_pred_conf_matrix = torch.argmax(y_pred, dim=1)
                y_true_conf_matrix = torch.argmax(y_true, dim=1)
            confusion_matrix = confusion_matrix_metric(y_pred_conf_matrix, y_true_conf_matrix)

            # TODO: encontrar uma forma melhor de fazer esse reshape
            y_true_roc = y_true.to(torch.int32)
            if self._number_of_outputs == 6:
                y_true_roc = torch.argmax(y_true_roc, dim=1)
            fpr, tpr, thresholds = roc_metric(y_pred, y_true_roc)

            if self._number_of_outputs == 6:
                self._fpr_multiclass = fpr.T.cpu().numpy()
                self._tpr_multiclass = tpr.T.cpu().numpy()
                self._thresholds_multiclass = thresholds.T.cpu().numpy()
            else:
                roc_metrics = torch.cat((fpr.reshape(-1, 1), tpr.reshape(-1, 1), thresholds.reshape(-1, 1)), dim=1)
                self._roc_metrics = roc_metrics.cpu().numpy()

            if self._feature_generator == 'CNNIDSFeatureGenerator':
                dummy_input = torch.randn(1, 1, self._window_size, self._number_of_bytes*2, dtype=torch.float).to(device)
            else:
                dummy_input = torch.randn(1, 1, self._window_size, self._number_of_bytes, dtype=torch.float).to(device)

            if device.type == "cpu":
                print("detection time in cpu")
                timing_func = timing.pytorch_inference_time_cpu
            else:
                print("detection time in gpu")
                timing_func = timing.pytorch_inference_time_gpu
            inference_time = timing_func(self._model, dummy_input)
            inference_time = inference_time / len(dummy_input)

            # TODO: Change this to be only used in case model is random forest
            # model_size = storage.pytorch_compute_model_size_mb(self._model)

            # Append metrics on list
            self._evaluation_metrics.append([fold, acc, prec, recall, f1, roc_auc, inference_time, self._threshold])
            self._confusion_matrix = confusion_matrix.cpu().numpy()

            if self._number_of_outputs == 1:
                fpr, tpr, thresholds = roc_curve(y_true.tolist(), y_pred.tolist())
                aucroc = roc_auc_score(y_true.tolist(), y_pred.tolist())
                plt.figure()
                plt.plot([0, 2], [0, 2], color="navy", lw=2, linestyle="--")
                plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {aucroc:.4f})")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                plt.savefig(f"{self._metrics_output_path}/roc_plot_{self._labeling_schema}_fold_{fold}_{self._model_name}.png", dpi=300, bbox_inches='tight')
                plt.close()


    def execute(self, data):
        def collate_gpu(batch):
            x, t = torch.utils.data.dataloader.default_collate(batch)
            return x.to(device="cuda:0"), t.to(device="cuda:0")
        # Reset all seed to ensure reproducibility
        self.__seed_all(0)
        g = torch.Generator()
        g.manual_seed(42)

        criterion_without_reduction = None
        if (self._criterion == 'mean-squared-error'):
            criterion_without_reduction = nn.MSELoss(reduction='none')
        elif (self._criterion == 'categorical-cross-entropy'):
            criterion_without_reduction = nn.CrossEntropyLoss(reduction='none')
        elif (self._criterion == 'mean-absolute-error'):
            criterion_without_reduction = nn.L1Loss(reduction='none')

        # Use gpu to train as preference
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fold_index = None

        # for fold_index in self._presaved_models_state_dict.keys():
        print('------------fold no---------{}----------------------'.format(fold_index))

        testloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=self._batch_size,
                    generator=g,
                    worker_init_fn=self.__seed_worker,
                    collate_fn=collate_gpu)

        print(f"len(testloader) = {len(testloader)}")

        for presaved_key in self._presaved_models_state_dict.keys():
            fold_index = presaved_key
            self._model.load_state_dict(torch.load(self._presaved_models_state_dict[presaved_key], map_location='cpu'))

            self._model.to(device)

            # This is only used in case you want to generate data for random forest models
            # self.__model_cnn_forward(device, testloader, fold_index)

            # Perform test step
            self.__test_model(device, testloader, fold_index, criterion_without_reduction)

            # Export metrics
            metrics_df = pd.DataFrame(self._evaluation_metrics, columns=["fold", "acc", "prec", "recall", "f1", "roc_auc", "inference_time", "threshold"])
            metrics_df.to_csv(f"{self._metrics_output_path}/test_metrics_{self._labeling_schema}_{self._model_name}_BS{self._batch_size}_fold_{fold_index}.csv")
            confusion_matrix_df = pd.DataFrame(self._confusion_matrix)
            confusion_matrix_df.to_csv(f"{self._metrics_output_path}/confusion_matrix_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")

            if self._number_of_outputs == 6:
                tpr_df = pd.DataFrame(self._tpr_multiclass)
                fpr_df = pd.DataFrame(self._fpr_multiclass)
                thresholds_df = pd.DataFrame(self._thresholds_multiclass)

                tpr_df.to_csv(f"{self._metrics_output_path}/tpr_multiclass_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
                fpr_df.to_csv(f"{self._metrics_output_path}/fpr_multiclass_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
                thresholds_df.to_csv(f"{self._metrics_output_path}/thresholds_multiclass_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
            else:
                roc_metrics_df = pd.DataFrame(self._roc_metrics, columns=["fpr", "tpr", "thresholds"])
                roc_metrics_df.to_csv(f"{self._metrics_output_path}/roc_metrics_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")