import os
from instrumentation import install
install()
import e2fl.server_app_llm as implementation
implementation._LOCAL_PATH = os.environ['TTA_OUTPUT_DIR']
from run_settings import STATE_MODE, STATE_JSON
if STATE_MODE == 'reuse':
    import json
    from predictor_state import validate, restore
    original_create_strategy = implementation.create_strategy
    def create_strategy(context):
        strategy = original_create_strategy(context)
        return restore(strategy, validate(json.loads(STATE_JSON)))
    implementation.create_strategy = create_strategy
app = implementation.app
