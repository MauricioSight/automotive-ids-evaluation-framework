import os
import torch
import random
import typing
import datetime

import pandas as pd
import numpy as np

from torch import nn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

from . import abstract_model_train_validate
from custom_metrics import timing, storage

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)

class PytorchModelTrainValidation(abstract_model_train_validate.AbstractModelTrainValidate):
    def __init__(self, model, model_config_dict: typing.Dict, feat_gen_dict: typing.Dict):
        self._model = model

        self._model_name = model_config_dict['model_name']
        self._model_type = model_config_dict['model_type']
        self._criterion = model_config_dict['criterion']
        self._cross_validation_folds = model_config_dict['cross_validation_folds']
        self._model_specs_dict = model_config_dict

        hyperparameters_dict = model_config_dict.get('hyperparameters')
        self._learning_rate = hyperparameters_dict['learning_rate']
        self._batch_size = hyperparameters_dict['batch_size']
        self._num_epochs = hyperparameters_dict['num_epochs']

        self._window_size = feat_gen_dict['config']['window_size']
        self._number_of_bytes = feat_gen_dict['config']['number_of_bytes']
        self._feature_generator = feat_gen_dict['feature_generator']
        self._multiclass = feat_gen_dict['config']['multiclass']
        self._dataset = feat_gen_dict['config']['dataset']
        self._labeling_schema = feat_gen_dict['config']['labeling_schema']

        self._evaluation_metrics = []
        self._train_validation_losses = []

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_pytorch_train"
        self._models_output_path_from_cofig = model_config_dict['paths']['models_output_path']
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

        self._models_output_path = f"{self._artifacts_path}/models"
        if not os.path.exists(self._models_output_path):
            os.makedirs(self._models_output_path)
            print("Models output directory created successfully")

        self._early_stopping_patience = hyperparameters_dict['early_stopping_patience']
        self._best_val_loss = float("inf")
        self._epochs_without_improvement = 0

        self._number_of_outputs = hyperparameters_dict.get('num_outputs', -1)
        
        self._roc_metrics = None

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


    def __reset_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()


    def __save_model_state_dict(self, fold=None):
        self._model.eval()

        if fold is not None:
            output_filename = f"{self._models_output_path_from_cofig}/pytorch_model_{self._model_name}_{fold}"
            torch.save(self._model.state_dict(), output_filename)
            output_filename = f"{self._models_output_path}/{self._dataset}_pytorch_model_{self._model_name}_{fold}"
        else:
            output_filename = f"{self._models_output_path_from_cofig}/pytorch_model_{self._model_name}_entire_dataset"
            torch.save(self._model.state_dict(), output_filename)
            output_filename = f"{self._models_output_path}/pytorch_model_{self._model_name}_entire_dataset"

        torch.save(self._model.state_dict(), output_filename)

    def __check_early_stopping(self, val_loss) -> int:
        ret = 0
        # Early stopping update
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement = self._epochs_without_improvement + 1

        # Early stopping condition
        if self._epochs_without_improvement >= self._early_stopping_patience:
            ret = -1

        return ret

    def __reset_early_stopping(self):
        self._best_val_loss = float("inf")
        self._epochs_without_improvement = 0

    def __train_model(self, criterion, device, trainloader, fold, epoch, learning_rate=None) -> int:
        self._model.train()
        train_loss = 0

        self._model = self._model.to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=(learning_rate if learning_rate else self._learning_rate))

        if self._number_of_outputs > 1:
            accuracy_metric = MulticlassAccuracy(num_classes=self._number_of_outputs).to(device)
        else:
            accuracy_metric = BinaryAccuracy().to(device)

        for batch_idx, (data, target) in enumerate(trainloader):
            data = data.float()
            if len(target.shape) == 1:
                target = target.reshape(-1, 1)
            target = target.float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # # forward + backward + optimize
            # # TODO: Later think a way this to be included inside model
            # if (self._model_name == "MultiStageIDS"):
            #     # Run stages
            #     y1 = self._model.forward_first_stage(data)
            #     y2 = self._model.forward_second_stage(data)

            #     # Move to devices
            #     y1 = y1.to(device)
            #     y2 = y2.to(device)

            #     print(f"y1 = {y1}")
            #     print(f"y1.shape = {y1.shape}")
            #     print(f"y2 = {y2}")
            #     print(f"y2.shape = {y2.shape}")

            #     # Combine data
            #     data = torch.cat((y1, y2), axis=1)

            if self._model_type == 'anomaly_detection':
                model_X = data
                target = data
            
            elif self._model_type == 'classification':
                model_X = data
                target = target

            output = self._model(model_X)

            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1)

            ## update model params
            optimizer.step()

            if self._model_type == 'classification':
                output = output.detach().round()
                acc = accuracy_metric(output, target)

                ## metrics logs
                if batch_idx % 1000 == 0:
                    # accuracy = 100 * correct / len(trainloader)
                    print('Train Fold: {} \t Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(
                    fold,epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item(), acc))

            elif self._model_type == 'anomaly_detection':
                ## metrics logs
                if batch_idx % 1000 == 0:
                    print('Train Fold: {} \t Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    fold,epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))


        train_loss = train_loss / len(trainloader)

        return train_loss

    def __validate_model(self, criterion, device, testloader, fold, epoch) -> typing.Tuple[int, float]:
        self._model.eval()
        ret = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                # if (self._model_name == "MultiStageIDS"):
                #     # Run stages
                #     y1 = self._model.forward_first_stage(data)
                #     y2 = self._model.forward_second_stage(data)

                #     # Move to devices
                #     y1 = y1.to(device)
                #     y2 = y2.to(device)

                #     # Combine data
                #     data = torch.cat((y1, y2), axis=1)
                
                if self._model_type == 'anomaly_detection':
                    model_X = data
                    target = data
                
                elif self._model_type == 'classification':
                    model_X = data
                    target = target

                output = self._model(model_X)

                val_loss += criterion(output, target).item()  # sum up batch loss

        val_loss = val_loss / len(testloader)
        ret = self.__check_early_stopping(val_loss)

        print('Train Fold: {} \t Epoch: {} \tValidation Loss: {:.6f}'.format(
                    fold, epoch, val_loss))

        return ret, val_loss

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

            print(f'Shape of attack label {attack_label}: {y_t.shape}')

            y_t = y_t.replace({2: 1, 3: 1, 4: 1, 5: 1})

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

    def __test_model(self, criterion, device, testloader, fold, criterion_without_reduction):
        self._model.eval()

        if self._number_of_outputs > 1:
            accuracy_metric = MulticlassAccuracy(num_classes=self._number_of_outputs).to(device)
            f1_score_metric = MulticlassF1Score(num_classes=self._number_of_outputs).to(device)
            auc_roc_metric = MulticlassAUROC(num_classes=self._number_of_outputs).to(device)
            precision_score = MulticlassPrecision(num_classes=self._number_of_outputs).to(device)
            recall_score = MulticlassRecall(num_classes=self._number_of_outputs).to(device)
        else:
            accuracy_metric = BinaryAccuracy().to(device)
            f1_score_metric = BinaryF1Score().to(device)
            auc_roc_metric = BinaryAUROC().to(device)
            precision_score = BinaryPrecision().to(device)
            recall_score = BinaryRecall().to(device)

        threshold = None
        
        # TODO: alterar para prealocar y_pred e y_true pra exeutar mais rapido
        # n_entries = self._batch_size * len(testloader)
        # y_pred = torch.zeros((n_entries, self._number_of_outputs)).to(device)
        # y_true = torch.zeros((n_entries, self._number_of_outputs)).to(device)
        y_pred = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                # if (self._model_name == "MultiStageIDS"):
                #     # Run stages
                #     y1 = self._model.forward_first_stage(data)
                #     y2 = self._model.forward_second_stage(data)

                #     # Move to devices
                #     y1 = y1.to(device)
                #     y2 = y2.to(device)

                #     # Combine data
                #     data = torch.cat((y1, y2), axis=1)

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
            if self._model_type in ['anomaly_detection']:
                y_true = torch.where((y_true == 2) | (y_true == 3) | (y_true == 4) | (y_true == 5), torch.tensor(1), y_true)

                threshold = self._get_threshold_youden_index(y_true=y_true, y_pred=y_pred)
                output = y_pred > threshold

                auc_roc_metric.update(y_pred, y_true)
                accuracy_metric.update(output, y_true)
                f1_score_metric.update(output, y_true)
                precision_score.update(output, y_true)
                recall_score.update(output, y_true)

                self._get_metrics_per_attack(roc_df, fold, threshold)

            # Calculate metrics
            acc = accuracy_metric.compute().cpu().numpy()
            f1 = f1_score_metric.compute().cpu().numpy()
            roc_auc = auc_roc_metric.compute().cpu().numpy()
            prec = precision_score.compute().cpu().numpy()
            recall = recall_score.compute().cpu().numpy()

            if self._feature_generator == 'CNNIDSFeatureGenerator':
                dummy_input = torch.randn(1, 1, self._window_size, self._number_of_bytes*2, dtype=torch.float).to(device)
            else:
                dummy_input = torch.randn(1, 1, self._window_size, self._number_of_bytes, dtype=torch.float).to(device)

            if device.type == "cpu":
                timing_func = timing.pytorch_inference_time_cpu
            else:
                timing_func = timing.pytorch_inference_time_gpu
            inference_time = timing_func(self._model, dummy_input)

            # TODO: Change this to be only used in case model is random forest
            model_size = storage.pytorch_compute_model_size_mb(self._model)

            # Append metrics on list
            self._evaluation_metrics.append([fold, acc, prec, recall, f1, roc_auc, inference_time, model_size, threshold])

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


    def execute(self, train_data, batch_size=None, learning_rate=None):
        def collate_gpu(batch):
            x, t = torch.utils.data.dataloader.default_collate(batch)
            return x.to(device="cuda:0"), t.to(device="cuda:0")
            # return x.to(device="cpu"), t.to(device="cpu")

        # Reset all seed to ensure reproducibility
        self.__seed_all(0)
        g = torch.Generator()
        g.manual_seed(42)

        # Use gpu to train as preference
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        # Get this criterion from configuration parameter
        criterion = None
        if (self._criterion == 'binary-cross-entropy'):
            criterion = nn.BCELoss()
        elif (self._criterion == 'categorical-cross-entropy'):
            criterion = nn.CrossEntropyLoss()
        elif (self._criterion == 'mean-squared-error'):
            criterion = nn.MSELoss()
        elif (self._criterion == 'mean-absolute-error'):
            criterion = nn.L1Loss()
        else:
            raise KeyError(f"Selected criterion : {self._criterion} is NOT available!")

        criterion_without_reduction = None
        if (self._criterion == 'mean-squared-error'):
            criterion_without_reduction = nn.MSELoss(reduction='none')
        elif (self._criterion == 'categorical-cross-entropy'):
            criterion_without_reduction = nn.CrossEntropyLoss(reduction='none')
        elif (self._criterion == 'mean-absolute-error'):
            criterion_without_reduction = nn.L1Loss(reduction='none')
        elif (self._criterion == 'binary-cross-entropy'):
            criterion_without_reduction = nn.BCELoss(reduction='none')

        skf = StratifiedKFold(n_splits=self._cross_validation_folds, random_state=1, shuffle=True)

        # Get item from train data
        X = [item[0] for item in train_data]
        y = [item[1] for item in train_data]

        # TODO: Find a better way to do this validation
        # This is to check if y has more than one dimension, for the multiclass case to work with skf.split
        try:
            y = np.array(y).argmax(1)
        except:
            pass

        # UNCOMENT THIS PART TO TRAIN USING STRATIFIED KFOLD
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print('------------fold no---------{}----------------------'.format(fold))

            # Filter indices where y == 0 for training and validation
            benign_idx = [i for i in train_idx if y[i] == 0]

            train_idx, val_idx = train_test_split(benign_idx, test_size=0.2, random_state=10)

            # Ensure the test set contains all remaining indices
            # test_idx = list(set(test_idx).union(set(train_idx) - set(benign_idx)))

            print(f'Train subsampler shape: {len(train_idx)}')
            print(f'Val subsampler shape: {len(val_idx)}')
            print(f'Test subsampler shape: {len(test_idx)}')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx, generator=g)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx, generator=g)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx, generator=g)

            trainloader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=(batch_size if batch_size else self._batch_size),
                        sampler=train_subsampler,
                        generator=g,
                        worker_init_fn=self.__seed_worker,
                        collate_fn=collate_gpu)

            valloader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=(batch_size if batch_size else self._batch_size),
                        sampler=val_subsampler,
                        generator=g,
                        worker_init_fn=self.__seed_worker,
                        collate_fn=collate_gpu)

            testloader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=(batch_size if batch_size else self._batch_size),
                        sampler=test_subsampler,
                        generator=g,
                        worker_init_fn=self.__seed_worker,
                        collate_fn=collate_gpu)

            # TODO: adicionar o carregamento dos modelos
            self._model.apply(self.__reset_weights)
            if (self._model_name == "MultiStageIDS"):
                random_forest_path = self._model_specs_dict["first_stage"]["presaved_paths"][f"{fold}"]
                pruned_cnn_path = self._model_specs_dict["second_stage"]["presaved_paths"][f"{fold}"]
                self._model.load_stages_models(random_forest_path, pruned_cnn_path)

            for epoch in range(self._num_epochs):
                train_loss = self.__train_model(criterion, device, trainloader, fold, epoch, learning_rate)
                ret, val_loss = self.__validate_model(criterion, device, valloader, fold, epoch)
                if (ret < 0):
                    print(f"Early stopping! Validation loss hasn't improved for {self._early_stopping_patience} epochs")
                    break

                self._train_validation_losses.append([fold, epoch, train_loss, val_loss])

            print(">> Testing model...")
            self.__test_model(criterion, device, testloader, fold, criterion_without_reduction)

            # Reset early stopping for next fold
            self.__reset_early_stopping()

            # Save model
            self.__save_model_state_dict(fold)

            # Export metrics
            metrics_df = pd.DataFrame(self._evaluation_metrics, columns=["fold", "acc", "prec", "recall", "f1", "roc_auc", "inference_time", "model_size", "threshold"])
            metrics_df.to_csv(f"{self._metrics_output_path}/val_metrics_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")

            train_val_loss_df = pd.DataFrame(self._train_validation_losses, columns=["fold", "epoch", "train_loss", "val_loss"])
            train_val_loss_df.to_csv(f"{self._metrics_output_path}/train_val_losses_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")

            roc_metrics_df = pd.DataFrame(self._roc_metrics, columns=["fpr", "tpr", "thresholds"])
            roc_metrics_df.to_csv(f"{self._metrics_output_path}/roc_metrics_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}_fold_{fold}.csv")

            # Plotting the losses
            plt.figure(figsize=(10, 6))

            # TODO: UNCOMMENT THIS
            # Group by fold and plot each fold's training and validation losses
            # for fold, group in train_val_loss_df.groupby('fold'):
            #     plt.plot(group['epoch'], group['train_loss'], label=f'Train Loss (Fold {fold})', linestyle='-')
            #     plt.plot(group['epoch'], group['val_loss'], label=f'Val Loss (Fold {fold})', linestyle='--')

            # Adding plot details
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss per Epoch')
            plt.legend()
            plt.grid(True)

            # Save the plot to a file
            plt.savefig(f'{self._metrics_output_path}/train_val_losses_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.png', dpi=300, bbox_inches='tight')
            plt.close()
