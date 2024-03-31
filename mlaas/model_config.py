import models.svm as svm
import models.logisticregression as logisticregression
import models.cnn as cnn
import models.fullconnect as fullconnect


GENERAL_REQUIRED_PARAMS = {
    'lr': 'float',
    'iterations': 'int',
    'batch_size': 'int',
    'data': 'file',
}

SPECIAL_REQUIRED_PARAMS = {
    'SVM': {'input_dim': 'int', 'output_dim': 'int'},
    # 'LR': {'input_dim': 'int', 'output_dim': 'int'},
    'FullConnect': {'input_dim': 'int', 'n_hidden_1': 'int', 'n_hidden_2': 'int', 'output_dim': 'int'},
    # 'CNN': {},
}


ALL_MODEL_CLASSES = {
    'SVM': 'Support Vector Machine',
    # 'LR': 'Logistic Regression',
    'FullConnect': 'Full Connect Neural Network',
    # 'CNN': 'Convolutional Neural Network',
}


def get_model_creator(model_name: str):
    if model_name == 'SVM':
        return svm.create_SVM
    elif model_name == 'LR':
        return logisticregression.create_LogisticRegression
    elif model_name == 'CNN':
        return cnn.create_cnn
    elif model_name == 'FullConnect':
        return fullconnect.create_FullConnect
    else:
        return None
