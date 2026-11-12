import threading
import time

from mapper.server import GpuQueue, Mapper

from .conftest import FakeRunner


def test_first_come_first_served():
    gpu, order, started = GpuQueue(), [], []

    def job(n):
        with gpu.turn() as info:
            started.append((n, info["ahead_on_arrival"]))
            time.sleep(0.1)
            order.append(n)

    threads = []
    for n in range(5):
        threads.append(threading.Thread(target=job, args=(n,)))
        threads[-1].start()
        time.sleep(0.01)  # arrive in order, all before the first one is done
    for t in threads:
        t.join()
    assert order == [0, 1, 2, 3, 4]
    assert [ahead for _, ahead in started] == [0, 1, 2, 3, 4]
    assert gpu.state() == {"gpu_busy": False, "waiting": 0}


def test_state_while_busy():
    gpu, inside, release = GpuQueue(), threading.Event(), threading.Event()

    def hold():
        with gpu.turn():
            inside.set()
            release.wait()

    holder = threading.Thread(target=hold)
    holder.start()
    inside.wait()
    assert gpu.state() == {"gpu_busy": True, "waiting": 0}
    release.set()
    holder.join()


class SlowRunner(FakeRunner):
    def __call__(self, paths):
        time.sleep(0.05)
        return super().__call__(paths)


def test_others_are_visible(tmp_path):
    m = Mapper(SlowRunner(), tmp_path)
    jpeg = b"\xff\xd8"
    m.run("step session=alice", [jpeg, jpeg])
    bob = m.run("step session=bob", [jpeg, jpeg])
    assert bob["queue"]["other_sessions_active"] == 1
    assert "1 other session" in bob["queue"]["message"]
    assert m.run("queue", [])["queue"]["other_sessions_active"] == 2
    assert m.run("queue session=bob", [])["queue"]["other_sessions_active"] == 1


def test_waiting_is_reported(tmp_path):
    m = Mapper(SlowRunner(), tmp_path)
    jpeg = b"\xff\xd8"
    results = {}

    def send(name):
        results[name] = m.run(f"step session={name}", [jpeg, jpeg])

    threads = [threading.Thread(target=send, args=(n,)) for n in ("a", "b", "c")]
    for t in threads:
        t.start()
        time.sleep(0.01)
    for t in threads:
        t.join()
    waited = sorted(r["queue"]["ahead_on_arrival"] for r in results.values())
    assert waited == [0, 1, 2]
    assert max(r["queue"]["waited_s"] for r in results.values()) >= 0.05


def test_same_session_at_once_keeps_all_photos(tmp_path):
    m = Mapper(SlowRunner(), tmp_path)
    threads = [threading.Thread(target=m.run, args=("step session=shared", [bytes([i])] * 3)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    frames = sorted((tmp_path / "shared" / "frames").glob("*.jpg"))
    assert len(frames) == 12
    assert sorted(f.read_bytes() for f in frames) == sorted([bytes([i]) for i in range(4)] * 3)
