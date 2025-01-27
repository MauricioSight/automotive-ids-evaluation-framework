import argparse
import json

from feature_generator import (
    cnn_ids_feature_generator, 
    anomaly_detection_feature_generator,
)
from models import (
    conv_net_ids,
    multiclass_conv_net_ids,
    pruned_conv_net_ids,
    sklearn_classifier,
    cae_anomaly_detection,
    lstmae_anomaly_detection,
    seqwatch,
)
from model_train_validation import (
    pytorch_model_train_validate,
    sklearn_model_train_validate
)

AVAILABLE_FEATURE_GENERATORS = {
    "CNNIDSFeatureGenerator": cnn_ids_feature_generator.CNNIDSFeatureGenerator,
    "AnomalyDetectionFeatureGenerator": anomaly_detection_feature_generator.AnomalyDetectionFeatureGenerator,
}

AVAILABLE_IDS = {
    "CNNIDS": conv_net_ids.ConvNetIDS,
    "MultiClassCNNIDS": multiclass_conv_net_ids.MultiClassConvNetIDS,
    "PrunedCNNIDS": pruned_conv_net_ids.PrunedConvNetIDS,
    "SklearnClassifier": sklearn_classifier.SklearnClassifier,
    "CAEAnomalyDetector": cae_anomaly_detection.CAEAnomalyDetector,
    "LSTMAEAnomalyDetector": lstmae_anomaly_detection.LSTMAEAnomalyDetector,
    "SeqWatch": seqwatch.SeqWatch,
}

AVAILABLE_FRAMEWORKS = {
    "pytorch": pytorch_model_train_validate.PytorchModelTrainValidation,
    "sklearn": sklearn_model_train_validate.SklearnModelTrainValidation
}

def main():
    parser = argparse.ArgumentParser(description='Execute model train validation step')
    parser.add_argument('--model_train_valid_config', required=True, help='JSON File containing the config for the select model train validation procedure')
    args = parser.parse_args()

    try:
        with open(args.model_train_valid_config, 'r') as model_train_valid_config:
            model_train_valid_config_dict = json.load(model_train_valid_config)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")

    print("##### Loaded configuration fils #####")
    print(json.dumps(model_train_valid_config_dict, indent=4, sort_keys=True))

    feat_gen_config_dict = model_train_valid_config_dict['feat_gen']
    model_specs_dict = model_train_valid_config_dict['model_specs']

    feature_generator_name = feat_gen_config_dict['feature_generator']
    feature_generator_config = feat_gen_config_dict['config']
    feature_generator_load_paths = feat_gen_config_dict['load_paths']
    w_size = feature_generator_config['window_size']
    number_of_bytes = feature_generator_config['number_of_bytes']
    hidden_size = model_specs_dict['hyperparameters'].get('hidden_size', None)

    if feature_generator_name not in AVAILABLE_FEATURE_GENERATORS:
        raise KeyError(f"Selected feature generator: {feature_generator_name} is NOT available!")

    model_name = model_specs_dict['model']
    if model_name not in AVAILABLE_IDS:
        raise KeyError(f"Selected model: {model_name} is NOT available!")

    framework = model_specs_dict['framework']
    if framework not in AVAILABLE_FRAMEWORKS:
        raise KeyError(f"Selected framework: {framework} is NOT available!")

    print("> Loading features...")
    selected_feature_generator = AVAILABLE_FEATURE_GENERATORS[feature_generator_name](feature_generator_config)
    data = selected_feature_generator.load_features(feature_generator_load_paths)

    # TODO: Transformar isso numa função externa ao main
    print("> Creating model...")
    if framework == "pytorch":
        num_outputs = model_specs_dict.get('hyperparameters').get('num_outputs', 1)
        num_ensemble_inputs = model_specs_dict.get('hyperparameters').get('ensemble_inputs', 2)
        if model_name in ["CNNIDS", "PrunedCNNIDS", "MultiClassCNNIDS"]:
            if num_outputs > 1:
                model = AVAILABLE_IDS[model_name](number_of_outputs=num_outputs)
            else:
                model = AVAILABLE_IDS[model_name]()
        if model_name in ['CAEAnomalyDetector', 'LSTMAEAnomalyDetector', 'SeqWatch']:
            model = AVAILABLE_IDS[model_name](w_size=w_size, input_size=number_of_bytes, hidden_size=hidden_size)
        print(f">> {model_name} was created with {num_outputs} outputs")
    elif framework == "sklearn":
        model = AVAILABLE_IDS[model_name](model_specs_dict)

    print("> Initializing model training and evaluation...")

    print(model)
    train_validator = AVAILABLE_FRAMEWORKS[framework](model, model_specs_dict, feat_gen_config_dict)
    train_validator.execute(data)

    print("Model trained successfully!")


if __name__ == "__main__":
    main()
