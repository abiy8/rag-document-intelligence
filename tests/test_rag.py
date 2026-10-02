import pytest
from fastapi.testclient import TestClient
from app.engine import Engine, chunks
from app.main import create_app

def test_persistence_deduplication_and_deletion(tmp_path):
    e=Engine(str(tmp_path));d=e.ingest("policy", "Customer support tickets are retained for 90 days.")
    e.ingest("policy", "Customer support tickets are retained for 90 days.")
    assert e.client.count(e.collection).count==1
    e.close(); e=Engine(str(tmp_path))
    assert e.answer("How long are customer support tickets retained?")["sources"][0]["document_id"]==d["document_id"]
    e.delete(d["document_id"])
    assert e.answer("tickets")["abstained"]
    e.close()

def test_chunk_overlap_and_empty_document():
    assert chunks("a b c d e f",4,1)==["a b c d","d e f"]
    e=Engine(":memory:")
    with pytest.raises(ValueError): e.ingest("empty","  ")
    e.close()

def test_api_upload_auth_validation_and_sources(monkeypatch):
    monkeypatch.setenv("API_KEY","test-key")
    with TestClient(create_app(":memory:","hash")) as c:
        assert c.post("/documents",json={"name":"x","text":"y"}).status_code==401
        headers={"X-API-Key":"test-key"}
        assert c.post("/upload",headers=headers, files={"file":("a.exe",b"hello")}).status_code==415
        assert c.post("/upload",headers=headers, files={"file":("broken.pdf",b"invalid PDF")}).status_code==422
        assert c.post("/upload",headers=headers, files={"file":("policy.txt",b"Database backups run every six hours.")}).status_code==200
        r=c.post("/query",headers=headers,json={"question":"How often do database backups run?"}).json()
        assert "six hours" in r["answer"] and r["sources"][0]["name"]=="policy.txt"
        assert c.post("/query",headers=headers,json={"question":"q","k":99}).status_code==422

def test_ollama_generation_contract(monkeypatch):
    import httpx
    def post(self,url,**kwargs):
        assert kwargs["json"]["messages"][0]["role"]=="system"
        return httpx.Response(200,json={"message":{"content":"Tickets are retained for 90 days [1]."}},request=httpx.Request("POST",url))
    monkeypatch.setattr(httpx.Client,"post",post)
    e=Engine(":memory:",generation="ollama");e.ingest("policy","Tickets are retained for 90 days.")
    assert "[1]" in e.answer("tickets retention")["answer"]
    e.close()
