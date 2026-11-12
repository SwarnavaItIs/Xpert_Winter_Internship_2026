"""Transport capacity is separate from live inference qualification."""
import unittest
from tools.check_capacity import rehearsal


class CapacityTests(unittest.TestCase):
    def test_seventeen_thousand_original_messages_are_scanned_once_within_transport_limits(self):
        result=rehearsal()
        self.assertTrue(result['simulation_only'])
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['messages_sent_as_core'],17000)
        self.assertTrue(result['output_created'])
        self.assertLess(result['max_request_bytes'],120000)
        self.assertLess(result['reserved_units'],result['reservation_limit'])


if __name__=='__main__':unittest.main()
