import unittest
from decimal import Decimal

from app import pricing


class PricingTests(unittest.TestCase):
    def test_prepare_usage_log_mistral_per_token(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        result = pricing.prepare_usage_log("mistral", "mistral-medium-latest", usage)

        self.assertEqual(result["prompt_tokens"], 1000)
        self.assertEqual(result["completion_tokens"], 500)
        self.assertEqual(result["prompt_cost"], Decimal("0.002000"))
        self.assertEqual(result["completion_cost"], Decimal("0.002500"))
        self.assertEqual(result["total_cost"], Decimal("0.004500"))
        self.assertEqual(result["currency"], "USD")

    def test_prepare_usage_log_cohere_per_request(self):
        usage = {"requests": 2, "documents": 50}
        result = pricing.prepare_usage_log("cohere", "rerank-english-v3.0", usage)

        self.assertIsNone(result["prompt_tokens"])
        self.assertIsNone(result["completion_tokens"])
        self.assertEqual(result["prompt_cost"], Decimal("0.004000"))
        self.assertIsNone(result["completion_cost"])
        self.assertEqual(result["total_cost"], Decimal("0.004000"))
        self.assertEqual(result["currency"], "USD")


if __name__ == "__main__":
    unittest.main()
