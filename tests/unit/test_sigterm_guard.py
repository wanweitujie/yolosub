import json
from unittest.mock import patch
from train_v.process import base_train

def test_sigterm_guard_finished(monkeypatch, tmp_path):
    sf = tmp_path / "status.json"
    sf.write_text(json.dumps({"status": "finished"}), encoding="utf-8")
    monkeypatch.setattr(base_train, "STATUS_FILE", sf)
    with patch("train_v.process.base_train.signal.signal") as m_sig, \
         patch("train_v.process.base_train.sys.exit") as m_exit:
        base_train.setup_signal()
        handler = m_sig.call_args.args[1]
        handler(None, None)   # 补丁仍生效,覆盖 sys.exit
    assert json.loads(sf.read_text(encoding="utf-8"))["status"] == "finished"
    assert m_exit.call_count == 1

def test_sigterm_guard_running(monkeypatch, tmp_path):
    sf = tmp_path / "status.json"
    sf.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    monkeypatch.setattr(base_train, "STATUS_FILE", sf)
    with patch("train_v.process.base_train.signal.signal") as m_sig, \
         patch("train_v.process.base_train.sys.exit") as m_exit:
        base_train.setup_signal()
        handler = m_sig.call_args.args[1]
        handler(None, None)
    assert json.loads(sf.read_text(encoding="utf-8"))["status"] == "stopped"
    assert m_exit.call_count == 1
