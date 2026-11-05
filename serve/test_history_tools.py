"""#4663: a call to a tool that this turn's `tools` no longer lists, but that the conversation already called, is a tool call (not raw text).

    python -m unittest serve.test_history_tools
"""
import copy
import http.client
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import serve.server as server
from serve.frontend import OutputParser

TODO = {"name": "TodoWrite", "parameters": {"type": "object", "properties": {"todos": {"type": "array"}}}}
CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"


def history(*names):
    return {"messages": [{"role": "user", "content": "go"},
                         *[{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                           "function": {"name": n, "arguments": "{}"}}]} for i, n in enumerate(names)]]}


def run(tools, text=CALL):
    p = OutputParser(thinking=False, tools=tools, stream_tools=True, recover=True, declared_only=True)
    evs = p.feed(text) + p.finish("stop")
    return [e.call.name for e in evs if e.kind == "tool_call"], "".join(e.text for e in evs if e.kind == "content")


class HistoryTools(unittest.TestCase):
    def test_a_tool_the_conversation_called_is_a_tool_call_again(self):
        names, answer = run(server.with_history_tools([TODO], history("Bash")))
        self.assertEqual(names, ["Bash"])
        self.assertEqual(answer, "")

    def test_without_the_history_it_stays_text(self):
        names, answer = run(server.with_history_tools([TODO], history("Read")))
        self.assertEqual(names, [])
        self.assertIn("<tool_call>", answer)

    def test_declared_names_are_not_doubled_and_no_tools_stays_no_tools(self):
        tools = [TODO, {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
        self.assertIs(server.with_history_tools(tools, history("Bash", "TodoWrite")), tools)
        self.assertEqual(server.with_history_tools([], history("Bash")), [])
        self.assertIsNone(server.with_history_tools(None, history("Bash")))

    def test_anthropic_and_responses_history_are_not_extended(self):
        for req in ({"messages": [{"role": "assistant", "content": [{"type": "tool_use", "name": "Bash"}]}]},
                    {"input": [{"type": "function_call", "name": "Bash"}]}):
            self.assertIs(server.with_history_tools([TODO], req)[0], TODO)
            self.assertEqual(len(server.with_history_tools([TODO], req)), 1)

    def test_raw_chat_history_survives_sampling_merge_without_prompt_schemas(self):
        tok = server.ByteTokenizer()
        template = server.ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        svc = server.Service(server.MockEngine(tok, CALL), tok, template, sampling_defaults={"temperature": 0.4})
        svc.model_arch = "qwen4exp"
        svc.shared = {"top_p": 0.9}
        req = history("Bash")
        req.update(tools=[{"type": "function", "function": TODO}], reasoning_effort="none", max_tokens=256)
        before = copy.deepcopy(req)
        rendered_tools = []
        original = template.render
        def render(messages, tools=None, **kwargs):
            rendered_tools.append(copy.deepcopy(tools))
            return original(messages, tools=tools, **kwargs)
        with mock.patch.object(server.Service, "start_telemetry", return_value=None), mock.patch.object(template, "render", render):
            httpd = server.serve(svc, port=0)
            self.addCleanup(httpd.server_close)
            self.addCleanup(httpd.shutdown)
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
            try:
                conn.request("POST", "/v1/chat/completions", json.dumps(req), {"Content-Type": "application/json"})
                response = conn.getresponse()
                body = json.loads(response.read())
            finally:
                conn.close()
        self.assertEqual(response.status, 200, body)
        call = body["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("Bash", {"command": "ls -la"}))
        self.assertTrue(rendered_tools)
        self.assertTrue(all([t["name"] for t in tools] == ["TodoWrite"] for tools in rendered_tools))
        self.assertEqual(req, before)

    def test_non_chat_routes_ignore_extra_chat_history_fields(self):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, "</think>\n" + CALL), tok,
                             server.ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        with mock.patch.object(server.Service, "start_telemetry", return_value=None):
            httpd = server.serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        raw_history = history("Bash")["messages"]
        raw_history[-1]["content"] = "previous answer"
        raw_history.append({"role": "user", "content": "next"})
        for route in ("messages", "responses"):
            for stream in (False, True):
                with self.subTest(route=route, stream=stream):
                    body = {"messages": raw_history, "stream": stream}
                    if route == "messages":
                        body.update(max_tokens=256, tools=[{"name": "TodoWrite", "input_schema": TODO["parameters"]}])
                    else:
                        body.update(input="next", max_output_tokens=256,
                                    tools=[{"type": "function", **TODO}])
                    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
                    try:
                        conn.request("POST", "/v1/" + route, json.dumps(body), {"Content-Type": "application/json"})
                        response = conn.getresponse()
                        raw = response.read().decode()
                    finally:
                        conn.close()
                    self.assertEqual(response.status, 200, raw)
                    if stream:
                        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                        items = ([e["content_block"] for e in evs if e["type"] == "content_block_start"]
                                 if route == "messages" else [e["item"] for e in evs if e["type"] == "response.output_item.done"])
                    else:
                        result = json.loads(raw)
                        items = result["content" if route == "messages" else "output"]
                    self.assertEqual([i for i in items if i["type"] in ("tool_use", "function_call")], [])
                    if route == "messages":
                        answer = ("".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
                                  if stream else "".join(i.get("text", "") for i in items))
                    else:
                        answer = "".join(c.get("text", "") for i in items if i["type"] == "message" for c in i["content"])
                    self.assertIn("<function=Bash>", answer)

    def test_mcp_loop_does_not_augment_chat_shaped_history(self):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, "</think>\n" + CALL), tok,
                             server.ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        hub = SimpleNamespace(settings={"max_rounds": 1}, call=mock.Mock(side_effect=AssertionError("no MCP call")))
        events = list(server.run_with_mcp(svc, hub, [{"role": "user", "content": "next"}], [TODO], {},
                                          [11], True, 512, 512, history("Bash"), threading.Event(), set()))
        self.assertFalse([e for kind, e in events if kind == "event" and e.kind.startswith("tool_")])
        hub.call.assert_not_called()

    def test_odd_requests_are_left_alone(self):
        self.assertEqual(server.with_history_tools([TODO], None), [TODO])
        self.assertEqual(server.with_history_tools([TODO], {"messages": [None, {"role": "assistant", "tool_calls": [None, {}]}]}), [TODO])


if __name__ == "__main__":
    unittest.main()
