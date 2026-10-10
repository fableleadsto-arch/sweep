import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from sweep.desktop import local_stream


@pytest.mark.parametrize("text,expected", [
    ('{"mess', ''),
    ('{"message":"Hello', 'Hello'),
    ('{"message":"Line\\nnext\\', 'Line\nnext'),
    ('{"message":"Hello","proposal":{"capability":"computer.command"', 'Hello'),
    ('{"message":"\\u0048i', 'Hi'),
    ('{"message":"\\ud83d', '?'),
    ('{"message":"\\ud83d\\ude00', '\U0001f600'),
])
def test_json_preview_never_shows_control_fields_or_broken_escapes(text, expected):
    assert local_stream.preview_text(text, True) == expected


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def run_stream(monkeypatch, chunks, headers=None):
    sent, previews = [], []
    real_client = httpx.AsyncClient
    def reply(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, stream=Chunks(chunks), headers=headers)
    def client(**kwargs):
        assert kwargs['trust_env'] is False and kwargs['follow_redirects'] is False
        return real_client(**kwargs, transport=httpx.MockTransport(reply))
    monkeypatch.setattr(local_stream.httpx, 'AsyncClient', client)
    settings = SimpleNamespace(ollama_model='downloaded:latest', ollama_base_url='http://127.0.0.1:11434',
                               request_timeout_seconds=30)
    answer = asyncio.run(local_stream.stream_chat(settings, system='system',
        messages=[{'role': 'user', 'content': 'hello'}], json_mode=True, on_preview=previews.append))
    return answer, previews, sent


def test_split_stream_produces_previews_then_complete_answer(monkeypatch):
    chunks = [json.dumps({'response': '{"message":"Hel', 'done': False}).encode() + b'\n',
              json.dumps({'response': 'lo"}', 'done': True, 'eval_count': 4}).encode() + b'\n']
    wire = b''.join(chunks)
    answer, previews, sent = run_stream(monkeypatch, [wire[:9], wire[9:20], wire[20:]])
    assert answer.parsed == {'message': 'Hello'}
    assert previews[-1] == 'Hello'
    assert answer.completion_tokens == 4
    assert sent[0]['think'] is False and sent[0]['stream'] is True
    assert sent[0]['options']['num_predict'] == 600


@pytest.mark.parametrize('body', [
    b'{"response":"partial","done":false}\n',
    b'{"error":"engine stopped"}\n',
    b'[]\n',
    b'{"response":123,"done":true}\n',
    b'{"response":"","done":true}\n',
])
def test_incomplete_or_invalid_stream_never_claims_success(monkeypatch, body):
    with pytest.raises(ValueError):
        run_stream(monkeypatch, [body])


def test_stream_response_has_a_byte_limit(monkeypatch):
    monkeypatch.setattr(local_stream, 'MAX_RESPONSE_BYTES', 16)
    with pytest.raises(ValueError, match='size limit'):
        run_stream(monkeypatch, [b'x' * 17])


def test_truncated_response_keeps_answer_but_discards_proposals(monkeypatch):
    body = json.dumps({'response': '{"message":"A partial answer", "proposal":',
                       'done': True, 'done_reason': 'length'}).encode() + b'\n'
    answer, previews, _ = run_stream(monkeypatch, [body])
    assert answer.parsed['incomplete'] is True
    assert 'A partial answer' in answer.parsed['message']
    assert 'proposal' not in answer.parsed
    assert previews[-1] == 'A partial answer'
