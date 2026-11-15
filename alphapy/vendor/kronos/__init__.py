# NOTICE: vendored from https://github.com/shiyu-coder/Kronos (model/), commit
# 67b630e67f6a18c9e9be918d9b4337c960db1e9a, MIT licence (see LICENSE in this folder).
# Changes: kronos.py imports .module instead of model.module and no longer appends "../" to sys.path.

from .kronos import KronosTokenizer, Kronos, KronosPredictor

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


