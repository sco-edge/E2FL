"""Run-isolated exogenous Pi schedule for a future cross-client pilot.

The inherited estimators and native candidate search are not changed. This
module is synthetic-test only until a separate pilot is frozen and approved.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from e2fl.strategy.DynamicSizingPilot import PilotFrozenCost, PilotLastRate, PilotOnline


class PiSchedule:
    def update_train_config(self, server_round, train_config):
        if self.rebalancing_enabled is not False:
            raise RuntimeError('cross-client schedule requires rebalancing disabled')
        if self.static_allocation_after_round != 0:
            raise RuntimeError('original static-allocation flag freezes Pi; use zero')
        raw = str(train_config.get('external-pi-quota-schedule', '') or '')
        try:
            quotas = tuple(int(x.strip()) for x in raw.split(','))
        except ValueError as exc:
            raise ValueError('invalid external Pi schedule') from exc
        if not quotas or any(x < 1 for x in quotas):
            raise ValueError('external Pi schedule must contain positive quotas')
        if not 1 <= int(server_round) <= len(quotas):
            raise ValueError('round outside frozen Pi schedule')
        quota = quotas[int(server_round)-1]
        # Parent refreshes rpi_normal_quota from this key before predicting the
        # Pi reference or searching Jetson candidates.
        train_config['rpi-normal-quota'] = quota
        train_config['rpi-correction-max-quota'] = max(
            quota, int(train_config.get('rpi-correction-max-quota', quota)))
        result = super().update_train_config(server_round, train_config)
        if int(result['max_samples_per_round_RPi5']) != quota:
            raise RuntimeError('Pi request differs from exogenous schedule')
        if int(result.get('rpi-normal-quota', 0)) != quota:
            raise RuntimeError('Pi reference quota differs from schedule')
        pi_uid_samples = [int(v) for k,v in result.items()
                          if k.startswith('max_samples_per_round_uid_') and 'rpi' in k.lower()]
        if pi_uid_samples and any(v != quota for v in pi_uid_samples):
            raise RuntimeError('Pi UID request differs from exogenous schedule')
        self._schedule_record(server_round, result, quota)
        return result

    def _schedule_record(self, server_round, result, quota):
        record = {'round': int(server_round), 'policy': self.pilot_policy,
                  'scheduled_pi_quota': quota,
                  'reference_pi_quota': int(self.rpi_normal_quota),
                  'requested_pi_quota': int(result['max_samples_per_round_RPi5']),
                  'final_request': dict(result.items())}
        path = Path(self.log_dir) / f'cross_client_schedule_round_{int(server_round):02d}.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, default=str, sort_keys=True, indent=2) + '\n')
        os.replace(temp, path)


class ScheduledFrozenCost(PiSchedule, PilotFrozenCost):
    pilot_policy = 'A_scheduled_pi_frozen_jetson_cost'


class ScheduledLastRate(PiSchedule, PilotLastRate):
    pilot_policy = 'B_scheduled_pi_last_step_rate'


class ScheduledOnline(PiSchedule, PilotOnline):
    pilot_policy = 'C_scheduled_pi_original_online_cost'


class ScheduledFixed(PiSchedule, PilotOnline):
    """Original C calibration through round 4; freeze only Jetson S/H after it."""
    pilot_policy = 'F_scheduled_pi_fixed_jetson_after_round4'

    def update_train_config(self, server_round, train_config):
        result = super().update_train_config(server_round, train_config)
        if result.get('round_mode') != 'BOTH':
            raise RuntimeError('fixed calibration requires two-client normal round')
        keys = [k for k in result if k.startswith(
            ('max_samples_per_round_', 'local_steps_', 'local_epochs_'))
            and 'jetson' in k.lower()]
        if int(server_round) == 4:
            self._cross_client_fixed_jetson = {k: int(result[k]) for k in keys}
        elif int(server_round) > 4:
            frozen = getattr(self, '_cross_client_fixed_jetson', None)
            if not frozen:
                raise RuntimeError('missing real round-4 Jetson calibration')
            for key in keys:
                if key not in frozen:
                    raise RuntimeError(f'unexpected Jetson request key {key}')
            for key, value in frozen.items():
                result[key] = value
        decision_path = Path(self.log_dir) / f'pilot_decision_round_{int(server_round):02d}.json'
        decision = json.loads(decision_path.read_text())
        decision['native_config_before_fixed'] = decision['config_after']
        decision['config_after'] = dict(result.items())
        decision['fixed_jetson_override_applied'] = int(server_round) > 4
        temp = decision_path.with_suffix('.tmp')
        temp.write_text(json.dumps(decision, default=str, sort_keys=True, indent=2) + '\n')
        os.replace(temp, decision_path)
        # DecisionEvidence's native-search record is diagnostic; this separate
        # file identifies the actual dispatched request after the fixed override.
        self._schedule_record(server_round, result,
                              int(result['max_samples_per_round_RPi5']))
        return result
