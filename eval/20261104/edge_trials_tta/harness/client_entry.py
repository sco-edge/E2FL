import os
import e2fl.client_app_llm as implementation
from run_settings import STATE_MODE, PLANNED_SAMPLE_OFFSET, PLANNED_SAMPLE_COUNT
if hasattr(implementation, '_LOCAL_PATH'):
    implementation._LOCAL_PATH = os.environ['TTA_OUTPUT_DIR'] + '/' 
app = implementation.app

if STATE_MODE == 'prime' and PLANNED_SAMPLE_COUNT > 0:
    from sample_audit import record_planned_window
    _original_initialize_state = implementation.initialize_state

    def initialize_with_planned_inputs(context):
        result = _original_initialize_state(context)
        state = implementation._LOCAL_STATE
        if not state.get('tta_planned_inputs_recorded'):
            record_planned_window(
                state['trainloader'].dataset, state['partition_id'],
                state['device_name'], state['device_uid'],
                state['model_name'], state['dataset_name'],
                PLANNED_SAMPLE_OFFSET, PLANNED_SAMPLE_COUNT,
            )
            state['tta_planned_inputs_recorded'] = True
        return result

    implementation.initialize_state = initialize_with_planned_inputs

# Instrumentation guard: never silently train after dropping the server adapter.
_original_set_weights = implementation.set_weights
def checked_set_weights(net, arrays):
    if isinstance(arrays, dict) and not arrays:
        raise RuntimeError('TTA_INVALID_PAYLOAD: empty adapter after ArrayRecord conversion; no training was performed')
    return _original_set_weights(net, arrays)
implementation.set_weights = checked_set_weights
