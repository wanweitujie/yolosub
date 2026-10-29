"""夹具 smoke test:container 发现 + client 可达 + preload 可加载。"""
from conftest import load_preload


def test_container_alive(container, client):
    r = client.get("/gpu_status")
    assert r.status_code == 200
    assert r.json()["success"]

def test_preload_loadable(container):
    img = load_preload(container)
    assert len(img) > 0   # preload.webp 非空
