import unittest

from wifi_recovery_harness import evaluate_scenario


def node(node_id, *, status="active", uptime_ms=100_000, association_epoch=1, channel=1):
    return {
        "node_id": node_id,
        "status": status,
        "health": {
            "extended": True,
            "uptime_ms": uptime_ms,
            "association_epoch": association_epoch,
            "channel": channel,
        },
    }


def capture(at_ms, nodes, *, server_reachable=True, ap_uptime=None):
    return {
        "captured_at_unix_ms": at_ms,
        "server_reachable": server_reachable,
        "nodes": nodes,
        "ap_uptime_seconds": ap_uptime,
    }


class WifiRecoveryEvidenceTests(unittest.TestCase):
    def test_ap_outage_longer_than_initial_retry_window_recovers_without_node_reset(self):
        evidence = {
            "before": capture(0, [node(11), node(12), node(13)]),
            "during": capture(100_000, [node(11, status="stale"), node(12, status="stale"), node(13, status="stale")]),
            "after": capture(130_000, [
                node(11, uptime_ms=230_000, association_epoch=2),
                node(12, uptime_ms=230_000, association_epoch=2),
                node(13, uptime_ms=230_000, association_epoch=2),
            ]),
        }
        result = evaluate_scenario("ap-outage", evidence, expected_nodes={11, 12, 13}, minimum_outage_seconds=90)
        self.assertTrue(result["passed"])

    def test_ap_reboot_and_channel_change_requires_uptime_reset_and_new_channel(self):
        evidence = {
            "before": capture(0, [node(11), node(12), node(13)], ap_uptime=3_600),
            "during": capture(20_000, [
                node(11, status="stale"), node(12, status="stale"), node(13, status="stale"),
            ], server_reachable=True, ap_uptime=None),
            "after": capture(60_000, [
                node(11, uptime_ms=160_000, association_epoch=2, channel=6),
                node(12, uptime_ms=160_000, association_epoch=2, channel=6),
                node(13, uptime_ms=160_000, association_epoch=2, channel=6),
            ], ap_uptime=25),
        }
        result = evaluate_scenario("ap-reboot-channel", evidence, expected_nodes={11, 12, 13}, expected_channel=6)
        self.assertTrue(result["passed"])

    def test_helper_restart_or_closed_udp_keeps_association_epoch_and_node_uptime(self):
        evidence = {
            "before": capture(0, [node(11), node(12), node(13)]),
            "during": capture(30_000, [], server_reachable=False),
            "after": capture(45_000, [
                node(11, uptime_ms=145_000), node(12, uptime_ms=145_000), node(13, uptime_ms=145_000),
            ]),
        }
        result = evaluate_scenario("helper-restart", evidence, expected_nodes={11, 12, 13})
        self.assertTrue(result["passed"])

    def test_stale_to_active_rejects_a_usb_reset_disguised_as_recovery(self):
        evidence = {
            "before": capture(0, [node(11, uptime_ms=500_000)]),
            "during": capture(15_000, [node(11, status="stale", uptime_ms=515_000)]),
            "after": capture(30_000, [node(11, uptime_ms=5_000, association_epoch=1)]),
        }
        result = evaluate_scenario("stale-active", evidence, expected_nodes={11})
        self.assertFalse(result["passed"])
        self.assertIn("node 11 uptime moved backwards", result["failures"])


if __name__ == "__main__":
    unittest.main()
