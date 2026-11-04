import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import policy_sample_audit as audit


class SampleAuditTests(unittest.TestCase):
    def test_local_ids_are_mapped_and_repeated_rows_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';run.mkdir()
            manifest={'clients':{'0':{'train_indices':[100,101]},'1':{'train_indices':[200,201]}}}
            path=run/'partition_manifest.json';path.write_text(json.dumps(manifest))
            (run/'condition.json').write_text(json.dumps({'warmup_rounds':2,
                'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
            (run/'state.json').write_text(json.dumps({'status':'complete'}))
            work=[]
            for host,device,partition in (('rpi5','RPi5',1),('jetson','jetson_orin_nx',0)):
                directory=run/host/'output/test/clients/exposure_state/client'
                directory.mkdir(parents=True)
                (run/host/'supernode.job.json').write_text(json.dumps({'argv':['--node-config',f'partition-id={partition} num-partitions=2']}))
                for r in (1,3):
                    (directory/f'round_{r:04}.json').write_text(json.dumps({'round':r,
                        'visited_sample_ids':[0],'executed_examples':1,'eligible_unique_samples':2}))
                    work.append({'round':r,'device_name':device,'selected_samples':1})
            with (run/'policy_rounds.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(work[0]));writer.writeheader();writer.writerows(work)
            report=root/'report';report.mkdir()
            with (report/'prediction_check.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['round','device','predicted_s','actual_s','relative_error_pct','within_10pct'])
                writer.writeheader()
                for device,predicted,error,success in [('RPi5',10.5,5,True),('jetson_orin_nx',12,20,False)]:
                    writer.writerow(dict(round=3,device=device,predicted_s=predicted,actual_s=10,relative_error_pct=error,within_10pct=success))
            with patch.object(audit,'integrity',return_value=(True,'fixture')):
                result=audit.export(run,root/'report')
            self.assertEqual(result['cross_device_global_id_overlap'],0)
            for device in result['devices'].values():
                self.assertEqual(device['calibration_evaluation_overlap'],1)
                self.assertEqual(device['repeat_visits'],1)
            with (root/'report/sample_visits.csv').open() as stream:visits=list(csv.DictReader(stream))
            self.assertEqual({r['sample_global_id'] for r in visits},{'100','200'})
            self.assertEqual(len(visits),4)
            with (report/'sample_workload_audit.csv').open() as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(reader.fieldnames[4:10],['predicted_s','actual_train_time_s','absolute_error_s','relative_error_pct','prediction_success','one_minus_relative_error_pct'])
            self.assertEqual([r['evaluation_status'] for r in rows],['warmup_excluded']*2+['evaluated']*2)
            self.assertEqual(rows[2]['absolute_error_s'],'0.5')
            with (report/'prediction_accuracy_summary.csv').open() as stream:summary=list(csv.DictReader(stream))
            self.assertEqual(summary[0]['evaluated_device_rounds'],'2')
            self.assertEqual(float(summary[0]['prediction_accuracy_pct']),50)
            self.assertEqual(float(summary[0]['one_minus_mare_pct']),87.5)
            self.assertEqual(summary[0]['tta_pass'],'not_verified')
