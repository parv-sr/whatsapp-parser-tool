from django.test import SimpleTestCase

from apps.core.views import _infer_transaction_type, _with_metadata_defaults


class RagLogicTests(SimpleTestCase):
    def test_infer_transaction_type_distinguishes_lease_rent_sale(self):
            self.assertEqual(_infer_transaction_type("2bhk for rent in bandra"), "RENT")
            self.assertEqual(_infer_transaction_type("leave and license for 3 years"), "LEASE")
            self.assertEqual(_infer_transaction_type("ownership sale flat"), "SALE")

    def test_metadata_defaults_are_backfilled(self):
        result = _with_metadata_defaults({"location": None, "transaction_type": "", "contact_numbers": []})
        self.assertEqual(result["location"], "Not specified")
        self.assertEqual(result["transaction_type"], "UNKNOWN")
        self.assertEqual(result["building_name"], "Not specified")
        self.assertEqual(result["contact_numbers"], [])

        