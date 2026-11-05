"""Only complete, adjacent Chat tool-result ID runs are paired. All data is synthetic."""
import copy
import unittest

from serve.frontend import openai_to_messages


def assistant(ids):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "read", "arguments": "{}"}} for cid in ids]}


def result(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


class HistoryPairing(unittest.TestCase):
    def convert(self, messages):
        req = {"messages": messages}
        before = copy.deepcopy(req)
        converted, tools, _ = openai_to_messages(req)
        self.assertEqual(req, before)
        self.assertIsNone(tools)
        for message in converted:
            self.assertNotIn("_call_id", message)
            for call in message.get("tool_calls", []):
                self.assertNotIn("_call_id", call)
        return converted

    def test_complete_reversed_run_uses_assistant_call_order(self):
        messages = self.convert([assistant(["a", "b"]), result("b", "B"), result("a", "A")])
        self.assertEqual([m["content"] for m in messages[1:]], ["A", "B"])
        self.assertEqual([c["function"]["name"] for c in messages[0]["tool_calls"]], ["read", "read"])

    def test_partial_duplicate_mismatched_and_missing_ids_preserve_order(self):
        for calls, results in [(["a", "b"], ["b"]), (["a", "b"], ["b", "b"]),
                               (["a", "a"], ["b", "a"]), (["a", "b"], ["b", "c"]),
                               (["a", "b"], [None, "a"]), ([None, "b"], ["b", "a"]),
                               (["", "b"], ["b", ""]), ([1, "b"], ["b", 1])]:
            with self.subTest(calls=calls, results=results):
                messages = self.convert([assistant(calls), *[result(cid, str(i)) for i, cid in enumerate(results)]])
                self.assertEqual([m["content"] for m in messages[1:]], [str(i) for i in range(len(results))])

    def test_interrupted_or_nonassistant_run_is_not_reordered(self):
        for previous in ({"role": "user", "content": "hi"}, assistant(["a", "b"])):
            messages = [previous, result("b", "B"), {"role": "user", "content": "next"}, result("a", "A")]
            self.assertEqual([m["content"] for m in self.convert(messages)[1:]], ["B", "next", "A"])
        self.assertEqual([m["content"] for m in self.convert([result("b", "B"), result("a", "A")])], ["B", "A"])

    def test_mixed_text_and_image_parts_stay_with_their_result(self):
        image = [{"type": "text", "text": "B"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
        messages = self.convert([assistant(["a", "b"]), result("b", image), result("a", "A")])
        self.assertEqual(messages[1]["content"], "A")
        self.assertEqual(messages[2]["content"], [
            {"type": "text", "text": "B"}, {"type": "image", "source": "data:image/png;base64,AA=="}])


if __name__ == "__main__":
    unittest.main()
