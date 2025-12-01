# Note: actions, measures, obs_transformers, sensors require torch
# Only import them if torch is available in your environment
# from habitat_extensions import actions, measures, obs_transformers, sensors

from habitat_extensions.config.default import get_extended_config
# Import task to register VLN-CE datasets (VLNCEDatasetV1, RxRVLNCEDatasetV1)
from habitat_extensions import task
