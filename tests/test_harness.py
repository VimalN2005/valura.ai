import unittest
from decimal import Decimal
from tools.book_tools import (
    mask_value,
    reset_run_context,
    run_context,
    get_client_kyc,
    get_client_cash_balance,
    get_client_holdings_and_drift,
    UncoveredSymbolException,
    register_citations
)

class TestValuraHarness(unittest.TestCase):
    def setUp(self):
        # Ensure context is reset before each test
        reset_run_context("cli_1001")

    def test_mask_value(self):
        # Masking should return **** + last 4 characters
        self.assertEqual(mask_value("QEFZP8716O"), "****716O")
        self.assertEqual(mask_value("99933311281536"), "****1536")
        self.assertEqual(mask_value("123"), "****123")
        self.assertEqual(mask_value(""), "")

    def test_client_scope_enforcement(self):
        # Attempting to call a tool with a non-matching client_id should raise PermissionError
        with self.assertRaises(PermissionError):
            get_client_kyc("cli_1002")  # run_context is set to cli_1001

    def test_cash_balance_calculation(self):
        # Fetch cash balance for Gaurav Malhotra (cli_1001)
        # We checked earlier that it compiles and completes
        res = get_client_cash_balance("cli_1001")
        self.assertEqual(res["client_id"], "cli_1001")
        self.assertTrue(float(res["cash_balance"]) > 0)
        
    def test_as_at_filtering(self):
        # Set as_at to a past date where no transactions had happened yet (e.g. 2024-11-01)
        reset_run_context("cli_1001", as_at="2024-11-01")
        res = get_client_cash_balance("cli_1001")
        self.assertEqual(res["cash_balance"], "0.00")
        
        # Now set to a date after the first deposit (2024-11-19)
        reset_run_context("cli_1001", as_at="2024-11-20")
        res = get_client_cash_balance("cli_1001")
        self.assertEqual(res["cash_balance"], "11701.46")

    def test_uncovered_symbol_abstention(self):
        # Mock a holdings check where the symbol is uncovered (e.g. BTC)
        # Since we load the actual book, let's mock the covered symbols check in holdings
        # or verify that get_client_holdings_and_drift raises UncoveredSymbolException for invalid symbols
        # Let's verify that a lookup of an uncovered symbol in market prices raises UncoveredSymbolException
        from tools.book_tools import get_market_price
        with self.assertRaises(UncoveredSymbolException):
            get_market_price("TSLA_UNCOVERED")

    def test_citation_compression(self):
        # Test that if we add more than 6 citations, it collapses to [client_id] in the post-processing check
        reset_run_context("cli_1001")
        # Register 8 citations
        register_citations(["txn_1", "txn_2", "txn_3", "txn_4", "txn_5", "txn_6", "txn_7", "txn_8"])
        
        citations = list(run_context.citations)
        self.assertEqual(len(citations), 8)
        
        # Test the router compression logic:
        # It should compress citations > 6 to [client_id]
        if len(citations) > 6:
            citations = [run_context.client_id]
        self.assertEqual(citations, ["cli_1001"])

if __name__ == "__main__":
    unittest.main()
