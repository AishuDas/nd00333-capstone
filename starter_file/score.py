# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"gameId": pd.Series([0], dtype="int64"), "blueWardsPlaced": pd.Series([0], dtype="int64"), "blueWardsDestroyed": pd.Series([0], dtype="int64"), "blueFirstBlood": pd.Series([0], dtype="int64"), "blueKills": pd.Series([0], dtype="int64"), "blueDeaths": pd.Series([0], dtype="int64"), "blueAssists": pd.Series([0], dtype="int64"), "blueEliteMonsters": pd.Series([0], dtype="int64"), "blueDragons": pd.Series([0], dtype="int64"), "blueHeralds": pd.Series([0], dtype="int64"), "blueTowersDestroyed": pd.Series([0], dtype="int64"), "blueTotalGold": pd.Series([0], dtype="int64"), "blueAvgLevel": pd.Series([0.0], dtype="float64"), "blueTotalExperience": pd.Series([0], dtype="int64"), "blueTotalMinionsKilled": pd.Series([0], dtype="int64"), "blueTotalJungleMinionsKilled": pd.Series([0], dtype="int64"), "blueGoldDiff": pd.Series([0], dtype="int64"), "blueExperienceDiff": pd.Series([0], dtype="int64"), "blueCSPerMin": pd.Series([0.0], dtype="float64"), "blueGoldPerMin": pd.Series([0.0], dtype="float64"), "redWardsPlaced": pd.Series([0], dtype="int64"), "redWardsDestroyed": pd.Series([0], dtype="int64"), "redFirstBlood": pd.Series([0], dtype="int64"), "redKills": pd.Series([0], dtype="int64"), "redDeaths": pd.Series([0], dtype="int64"), "redAssists": pd.Series([0], dtype="int64"), "redEliteMonsters": pd.Series([0], dtype="int64"), "redDragons": pd.Series([0], dtype="int64"), "redHeralds": pd.Series([0], dtype="int64"), "redTowersDestroyed": pd.Series([0], dtype="int64"), "redTotalGold": pd.Series([0], dtype="int64"), "redAvgLevel": pd.Series([0.0], dtype="float64"), "redTotalExperience": pd.Series([0], dtype="int64"), "redTotalMinionsKilled": pd.Series([0], dtype="int64"), "redTotalJungleMinionsKilled": pd.Series([0], dtype="int64"), "redGoldDiff": pd.Series([0], dtype="int64"), "redExperienceDiff": pd.Series([0], dtype="int64"), "redCSPerMin": pd.Series([0.0], dtype="float64"), "redGoldPerMin": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
