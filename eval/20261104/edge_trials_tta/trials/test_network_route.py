import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import network_route as route


class NetworkRouteTests(unittest.TestCase):
    def test_restore_only_lease_owned_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            lease=Path(tmp)/'lease.json'
            current={'dev':route.INTERFACE,'prefsrc':route.SOURCE}
            calls=[]
            def fake_remote(cmd):
                calls.append(cmd)
                return '[{"dst":"192.168.128.124"}]' if cmd.startswith('ip -j route show') else ''
            with patch.object(route,'LEASE',lease),patch.object(route,'route',return_value=current),patch.object(route,'remote',side_effect=fake_remote):
                route.restore()
                self.assertFalse(calls)
                lease.write_text(json.dumps({'active':True,'interface':route.INTERFACE,
                                             'source':route.SOURCE}))
                route.restore()
                self.assertEqual(calls,[f'ip -j route show {route.DEST}/32',
                                        f'sudo -n ip route del {route.DEST}/32 dev {route.INTERFACE}'])
                self.assertFalse(json.loads(lease.read_text())['active'])

    def test_restore_marks_disappeared_route_inactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            lease=Path(tmp)/'lease.json'
            lease.write_text(json.dumps({'active':True,'interface':route.INTERFACE,
                                         'source':route.SOURCE}))
            with patch.object(route,'LEASE',lease), \
                 patch.object(route,'remote',return_value='[]') as remote, \
                 patch.object(route,'route',return_value={'dev':'enP8p1s0'}):
                route.restore()
                self.assertEqual(remote.call_args_list[0].args,
                                 (f'ip -j route show {route.DEST}/32',))
                self.assertFalse(json.loads(lease.read_text())['active'])
                self.assertIn('lost_at',json.loads(lease.read_text()))


if __name__=='__main__':unittest.main()
