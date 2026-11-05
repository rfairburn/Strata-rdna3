"""Offline passthrough tests with normal incremental tool-call streaming.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest serve.test_tool_call_passthrough -v
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from serve.frontend import OutputParser
from serve.server import tool_call_passthrough_setting
from serve import test_tool_routes as routes

TOOLS = [{"name": "bash", "parameters": {"properties": {
    "command": {"type": "string"}, "count": {"type": "integer"}, "enabled": {"type": "boolean"},
    "items": {"type": "array"}}}}]
WIDTHS = (0, 1, 2, 3, 5, 7, 17, 128)


def call(name="tools.bash", **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def parse(raw, width=0, *, reason="stop", **kw):
    opts = dict(thinking=False, tools=TOOLS, stream_tools=True, recover=True,
                declared_only=True, tool_call_passthrough=True)
    opts.update(kw)
    p = OutputParser(**opts)
    evs = []
    for at in range(0, len(raw), width or len(raw) or 1):
        evs += p.feed(raw[at:at + (width or len(raw))])
    evs += p.finish(reason)
    return evs, p


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs, kind="content"):
    return "".join(e.text or "" for e in evs if e.kind == kind)


class Parser(unittest.TestCase):
    def test_unknown_names_are_preserved_without_guessing(self):
        for name in ("tools.bash", "tools__bash", "tools=bash", "exec_command", "BASH", "unknown name", " unknown "):
            for width in WIDTHS:
                for recover in (False, True):
                    with self.subTest(name=name, width=width, recover=recover):
                        evs, p = parse(call(name, command="sample"), width, recover=recover)
                        self.assertEqual(calls(evs), [(name, {"command": "sample"})])
                        self.assertEqual(text(evs), "")
                        self.assertEqual((p.unread, p.unread_names), (0, []))
                        starts = [e.call.name for e in evs if e.kind == "tool_start"]
                        self.assertEqual(starts, [name])
                        args = "".join(e.text for e in evs if e.kind == "tool_args")
                        self.assertEqual(json.loads(args), {"command": "sample"})

    def test_names_and_string_arguments_stream_before_the_call_closes(self):
        for name in ("bash", "tools.bash"):
            for recover in (False, True):
                p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=recover,
                                 declared_only=True, tool_call_passthrough=True)
                evs = []
                for c in f"<tool_call>\n<function={name}>\n":
                    evs += p.feed(c)
                self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], [name])
                self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"), "{")
                evs += p.feed("<parameter=command>\nfirst part")
                self.assertEqual(calls(evs), [])
                if name == "bash":
                    self.assertIn("first part", "".join(e.text for e in evs if e.kind == "tool_args"))
                # An undeclared parameter keeps schema-less value decoding, so its value waits for </parameter>.
                evs += p.feed("\n</parameter>\n</function>\n</tool_call>") + p.finish()
                self.assertEqual(calls(evs), [(name, {"command": "first part"})])

    def test_declared_types_and_argument_repairs_are_retained(self):
        raw = call("bash", command="true", count="3", enabled="false", items="[1,]")
        for width in WIDTHS:
            for stream in (False, True):
                evs, _ = parse(raw, width, stream_tools=stream)
                self.assertEqual(calls(evs), [("bash", {"command": "true", "count": 3, "enabled": False, "items": [1]})])
                self.assertEqual(text(evs), "")

    def test_unknown_values_use_existing_schema_less_decoding(self):
        for width in WIDTHS:
            evs, _ = parse(call(command="true", count="3"), width)
            self.assertEqual(calls(evs), [("tools.bash", {"command": True, "count": 3})])

    def test_exact_names_and_ambiguous_aliases(self):
        tools = [{"name": n, "parameters": {"properties": {"command": {"type": "string"}}}}
                 for n in ("one.bash", "two.bash", "one__bash")]
        for name, want in (("one__bash", "one__bash"), ("one=bash", "one.bash"), ("bash", "bash")):
            for width in WIDTHS:
                evs, _ = parse(call(name, command="sample"), width, tools=tools)
                self.assertEqual(calls(evs), [(want, {"command": "sample"})])

    def test_fenced_and_inline_code_examples_remain_text(self):
        for name in ("bash", "tools.bash"):
            raw = call(name, command="sample")
            for example in ("```xml\n" + raw + "\n```", "~~~\n" + raw + "\n~~~", "`" + raw + "`"):
                for width in WIDTHS:
                    evs, _ = parse(example, width)
                    self.assertFalse([e for e in evs if e.kind.startswith("tool_")])
                    self.assertEqual(text(evs), example)

    def test_nested_calls_remain_parameter_data(self):
        value = "first\n" + call("tools.inner", command="sample") + "\nlast"
        for name in ("bash", "tools.bash"):
            for width in WIDTHS:
                evs, _ = parse(call(name, command=value), width)
                self.assertEqual(calls(evs), [(name, {"command": value})])
                self.assertEqual(text(evs), "")

    def test_multiple_calls_keep_normal_streaming(self):
        raw = "Before.\n" + call(command="sample") + "\n" + call("bash", command="other") + "\nAfter."
        for width in WIDTHS:
            evs, _ = parse(raw, width)
            self.assertEqual(calls(evs), [("tools.bash", {"command": "sample"}), ("bash", {"command": "other"})])
            self.assertEqual(text(evs), "Before.After.")
            self.assertEqual(len([e for e in evs if e.kind == "tool_start"]), 2)

    def test_shared_wrappers_keep_existing_recovery(self):
        raw = call(command="sample").removesuffix("\n</tool_call>") + "\n" + call("bash", command="other").removeprefix("<tool_call>\n")
        for width in WIDTHS:
            evs, _ = parse(raw, width)
            self.assertEqual(calls(evs), [("tools.bash", {"command": "sample"}), ("bash", {"command": "other"})])

    def test_missing_wrapper_close_keeps_natural_stop_recovery(self):
        for name in ("bash", "tools.bash"):
            for width in WIDTHS:
                raw = call(name, command="sample").removesuffix("\n</tool_call>")
                evs, _ = parse(raw, width)
                self.assertEqual(calls(evs), [(name, {"command": "sample"})])
                evs, _ = parse(raw, width, reason="length")
                self.assertEqual(calls(evs), [])
                self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], [name])

    def test_length_cut_can_announce_but_does_not_finalize_a_partial_call(self):
        for name in ("bash", "tools.bash"):
            raw = f"<tool_call>\n<function={name}>\n<parameter=command>\nfirst part"
            for width in WIDTHS:
                evs, _ = parse(raw, width, reason="length")
                self.assertEqual(calls(evs), [])
                self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], [name])
                self.assertTrue([e for e in evs if e.kind == "tool_args"])

    def test_unknown_calls_with_no_declared_tools(self):
        for tools in (None, []):
            for declared_only in (False, True):
                for width in WIDTHS:
                    evs, _ = parse(call(command="sample"), width, tools=tools, declared_only=declared_only)
                    self.assertEqual(calls(evs), [("tools.bash", {"command": "sample"})])

    def test_reasoning_calls_keep_quote_and_turn_end_guards(self):
        raw = "Plan.\n" + call(command="sample")
        for width in WIDTHS:
            for tools in (TOOLS, None):
                evs, _ = parse(raw, width, tools=tools, thinking=True)
                self.assertEqual(calls(evs), [("tools.bash", {"command": "sample"})])
                for suffix, reason in (("\nAn example.", "stop"), ("", "length"), ("", "error"), ("", "cancelled")):
                    evs, _ = parse(raw + suffix, width, tools=tools, thinking=True, reason=reason)
                    self.assertFalse([e for e in evs if e.kind.startswith("tool_")])
                    self.assertEqual(text(evs, "reasoning"), raw + suffix)

    def test_declared_bare_function_recovery_is_not_disabled(self):
        raw = call("bash", command="sample").removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
        for width in WIDTHS:
            evs, _ = parse(raw, width)
            self.assertEqual(calls(evs), [("bash", {"command": "sample"})])

    def test_json_wrapper_recovery_is_retained(self):
        raw = '<tool_call>{"name":"tools.bash","arguments":{"command":"sample"}}</tool_call>'
        for width in WIDTHS:
            evs, _ = parse(raw, width)
            self.assertEqual(calls(evs), [("tools.bash", {"command": "sample"})])

    def test_missing_function_close_keeps_typed_stream_final_arguments_and_ids(self):
        cases = (("integer", "30", 30), ("boolean", "false", False), ("array", "[1,]", [1]),
                 (None, "3", 3), (None, "true", True), (None, '{"a":false}', {"a": False}))
        for declared, value, want in cases:
            raw = '<tool_call><function=run_command><parameter=timeout>\n' + value + '\n</parameter></tool_call>'
            tools = [{"name": "run_command", "parameters": {"properties": {
                "timeout": {"type": declared} if declared else {}}}}]
            pieces = [[raw], *[[raw[:at], raw[at:]] for at in range(1, len(raw))], list(raw)]
            for recover in (False, True):
                for split, chunks in enumerate(pieces):
                    with self.subTest(declared=declared, value=value, recover=recover, split=split):
                        parser = OutputParser(thinking=False, tools=tools, stream_tools=True,
                                              recover=recover, declared_only=True)
                        evs = []
                        for chunk in chunks:
                            evs += parser.feed(chunk)
                        evs += parser.finish("stop")
                        final = [e.call for e in evs if e.kind == "tool_call"]
                        starts = [e.call for e in evs if e.kind == "tool_start"]
                        self.assertEqual([c.arguments for c in final], [{"timeout": want}])
                        self.assertEqual([c.id for c in starts], [c.id for c in final])
                        self.assertEqual(json.loads("".join(e.text for e in evs if e.kind == "tool_args")),
                                         final[0].arguments)
                        self.assertTrue(all(e.call.id == final[0].id for e in evs if e.kind == "tool_args"))

    def test_raw_name_whitespace_keeps_the_existing_reasoning_stop_edge(self):
        for width in WIDTHS:
            raw = call(" unknown name ", command="sample")
            evs, _ = parse(raw, width, thinking=True)
            self.assertEqual(calls(evs), [(" unknown name ", {"command": "sample"})])
            evs, _ = parse(raw.removesuffix("\n</tool_call>"), width, thinking=True)
            self.assertEqual(calls(evs), [("unknown name", {"command": "sample"})])
            for reason in ("length", "cancel", "error"):
                evs, _ = parse(raw.removesuffix("\n</tool_call>"), width, thinking=True, reason=reason)
                self.assertFalse([e for e in evs if e.kind.startswith("tool_")])

    def test_recovery_false_does_not_enable_json_or_parameter_openers(self):
        forms = ('<tool_call>{"name":"bash","arguments":{"command":"sample"}}</tool_call>',
                 call("bash", command="sample").replace("<function=bash>", "<parameter=bash>"),
                 call("bash", command="sample").removeprefix("<tool_call>\n").removesuffix("\n</tool_call>"))
        for raw in forms:
            for width in WIDTHS:
                evs, _ = parse(raw, width, recover=False)
                self.assertFalse([e for e in evs if e.kind.startswith("tool_")])
                evs, _ = parse(raw, width, recover=True)
                self.assertEqual(calls(evs), [("bash", {"command": "sample"})])

    def test_false_and_default_retain_declared_name_filtering(self):
        raw = call(command="sample")
        for width in WIDTHS:
            evs, p = parse(raw, width, tool_call_passthrough=False)
            self.assertFalse([e for e in evs if e.kind.startswith("tool_")])
            self.assertEqual((text(evs), p.unread, p.unread_names), (raw, 1, ["tools.bash"]))
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, declared_only=True)
        self.assertEqual(text(p.feed(raw) + p.finish()), raw)


class Settings(unittest.TestCase):
    def test_only_boolean_opt_in_is_accepted(self):
        self.assertIs(tool_call_passthrough_setting({}), False)
        self.assertIs(tool_call_passthrough_setting({"tool_call_passthrough": False}), False)
        self.assertIs(tool_call_passthrough_setting({"tool_call_passthrough": True}), True)
        for value in (None, 0, 1, "true", "false", [], {}):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                tool_call_passthrough_setting({"tool_call_passthrough": value})


class APIs(unittest.TestCase):
    def setUp(self):
        sampler = patch.object(routes.Service, "start_telemetry", return_value=None)
        sampler.start()
        self.addCleanup(sampler.stop)

    def every_route(self, script, want, answer="", enabled=True):
        for (route, stream), (finish, found, content) in routes.Routes().read(script, tool_call_passthrough=enabled).items():
            with self.subTest(route=route, stream=stream):
                self.assertEqual(found, want)
                self.assertEqual(content, answer)
                if route != "responses":
                    self.assertEqual(finish, ("tool_calls" if route == "chat" else "tool_use") if want
                                     else ("stop" if route == "chat" else "end_turn"))

    def test_unknown_name_reaches_all_six_api_paths(self):
        self.every_route("Plan.\n</think>\n" + call("tools.write_file", path="sample.txt"), [("tools.write_file", {"path": "sample.txt"})])

    def test_declared_tool_keeps_string_typing(self):
        self.every_route("Plan.\n</think>\n" + call("write_file", path="true"), [("write_file", {"path": "true"})])

    def test_natural_stop_wrapper_recovery_is_retained(self):
        raw = call("tools.write_file", path="sample.txt").removesuffix("\n</tool_call>")
        self.every_route("Plan.\n</think>\n" + raw, [("tools.write_file", {"path": "sample.txt"})])

    def test_fenced_examples_stay_text(self):
        raw = "```xml\n" + call("tools.write_file", path="sample.txt") + "\n```"
        self.every_route("Plan.\n</think>\n" + raw, [], raw)

    def test_default_route_policy_is_retained_when_disabled(self):
        raw = call("tools.write_file", path="sample.txt")
        self.every_route("Plan.\n</think>\n" + raw, [], raw, enabled=False)


if __name__ == "__main__":
    unittest.main()
