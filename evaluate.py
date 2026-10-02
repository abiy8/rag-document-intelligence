import argparse
import json
import time
from pathlib import Path
import numpy as np
from app.engine import Engine

def evaluate(backend="hash"):
    engine=Engine(":memory:", backend)
    docs=json.loads(Path("examples/documents.json").read_text())
    cases=json.loads(Path("examples/evaluation.json").read_text())
    for doc in docs:
        engine.ingest(doc["name"], doc["text"])
    rows=[]
    for case in cases:
        t=time.perf_counter(); result=engine.answer(case["question"], 3); elapsed=(time.perf_counter()-t)*1000
        ranks=[i for i,s in enumerate(result["sources"],1) if s["name"]==case["document"]]
        rows.append({"question":case["question"], "answerable":case["document"] is not None, "rank":min(ranks) if ranks else None, "answer_contains_expected":bool(case["answer"] and case["answer"].lower() in result["answer"].lower()), "abstained":result["abstained"], "latency_ms":elapsed})
    answerable=[r for r in rows if r["answerable"]]
    report={"backend":backend, "generation":"extractive", "dataset":"13 authored policy questions, 6 short synthetic documents; development benchmark, not a held-out real-world corpus", "recall_at_3":sum(r["rank"] is not None for r in answerable)/len(answerable), "mrr_at_3":sum(1/r["rank"] if r["rank"] else 0 for r in answerable)/len(answerable), "answer_contains_expected_rate":sum(r["answer_contains_expected"] for r in answerable)/len(answerable), "unanswerable_abstention_rate":sum(r["abstained"] for r in rows if not r["answerable"])/sum(not r["answerable"] for r in rows), "p95_query_ms":float(np.percentile([r["latency_ms"] for r in rows],95)), "cases":rows}
    engine.close();return report

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--backend", choices=["hash","semantic"],default="hash");a=p.parse_args()
    report=evaluate(a.backend);Path("reports").mkdir(exist_ok=True);Path(f"reports/evaluation-{a.backend}.json").write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="cases"},indent=2))
