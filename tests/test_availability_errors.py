"""Provider outages must still fit the backend availability API contract."""
import json
import unittest
from unittest import mock

import bot


class AvailabilityErrorTests(unittest.TestCase):
    def test_html_gateway_error_uses_status_instead_of_response_page(self):
        response = bot.requests.Response()
        response.status_code = 504
        response.reason = "Gateway Timeout"
        response._content = b"<!DOCTYPE html><html>" + b"upstream outage " * 1000 + b"</html>"
        error = bot.requests.HTTPError(response=response)
        self.assertEqual(bot.describe_http_error(error), "http_504: Gateway Timeout")

    def test_long_json_and_connection_errors_are_single_line_and_bounded(self):
        response = bot.requests.Response()
        response.status_code = 429
        response._content = json.dumps({"error": {"type": "quota", "code": "rate_limit",
                                                   "message": "try later\n" * 200}}).encode()
        reason = bot.describe_http_error(bot.requests.HTTPError(response=response))
        self.assertTrue(reason.startswith("http_429: quota: rate_limit:"))
        self.assertEqual(len(reason), 500)
        self.assertNotIn("\n", reason)
        reason = bot.describe_http_error(bot.requests.ConnectionError("connection failed\n" * 200))
        self.assertEqual(len(reason), 500)
        self.assertNotIn("\n", reason)

    def test_availability_report_bounds_reason_and_throttles_normalized_repeats(self):
        reason = "http_504: " + "upstream outage\n" * 200
        with mock.patch.dict(bot._MODEL_AVAILABILITY_REPORT_CACHE,
                             {"ready": None, "reason": "", "reported_at": 0.0}, clear=True):
            with mock.patch.object(bot, "post_json", return_value={"ok": True}) as post:
                with mock.patch.object(bot.time, "time", side_effect=[100.0, 101.0]):
                    with mock.patch.object(bot, "model_availability_report_interval_seconds", return_value=30.0):
                        self.assertTrue(bot.report_model_availability(False, reason))
                        self.assertFalse(bot.report_model_availability(False, reason + "different tail"))
                post.assert_called_once()
                route, payload = post.call_args.args
                self.assertEqual(route, "/bots/availability")
                self.assertIs(payload["ready"], False)
                self.assertEqual(payload["reason"], " ".join(reason.split())[:500])
                self.assertEqual(len(payload["reason"]), 500)
                self.assertNotIn("\n", payload["reason"])
