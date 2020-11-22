import numpy as np

np.random.seed(0)

from models.gen_models.model_inte import Integrate_Model_Cls_Ensemble_CAM_Att, Integrate_Model_Seg_Ensemble_Fusion

MODELS = {
    'Integrate_Model_Cls_Ensemble_CAM_Att': Integrate_Model_Cls_Ensemble_CAM_Att,
    'Integrate_Model_Seg_Ensemble_Fusion': Integrate_Model_Seg_Ensemble_Fusion
}
