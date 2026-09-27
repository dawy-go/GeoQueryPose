import os
import urllib.request
import urllib.parse
import urllib.error

BASE_URL = os.environ.get("GEOQUERYPOSE_PUSH_URL", "")
NTFY_URL = os.environ.get("GEOQUERYPOSE_NTFY_URL", "")
NTFY_TOPIC = os.environ.get("GEOQUERYPOSE_NTFY_TOPIC", "")
NTFY_TOKEN = os.environ.get("GEOQUERYPOSE_NTFY_TOKEN", "")


def push_notify(base_url=None, status: str = "up", msg: str = None, ping: int = None):
    base_url = base_url or BASE_URL
    if not base_url:
        return
    params = {'status': status}
    if msg is not None:
        params['msg'] = str(msg)
    if ping is not None:
        params['ping'] = int(ping)

    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    try:
        with urllib.request.urlopen(url, timeout=3):
            pass
    except urllib.error.HTTPError as e:
        if e.code == 502:
            print("Server returned 502 Bad Gateway, ignored. Program continues...")
        else:
            print(f"Encountered other HTTP error: {e.code}")
    except urllib.error.URLError as e:
        print(f"Network connection failed: {e.reason}")
    except Exception:
        pass


def push_ntfy(message, title=None, tags=None, priority=None,
              url=NTFY_URL, topic=NTFY_TOPIC, token=NTFY_TOKEN):
    if not url or not topic:
        return
    endpoint = f"{url.rstrip('/')}/{topic.lstrip('/')}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if title is not None:
        headers["Title"] = str(title)
    if tags is not None:
        headers["Tags"] = ",".join(tags) if isinstance(tags, (list, tuple)) else str(tags)
    if priority is not None:
        headers["Priority"] = str(priority)

    request = urllib.request.Request(
        endpoint,
        data=str(message).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3):
            pass
    except urllib.error.HTTPError as exc:
        print(f"ntfy HTTP error: {exc.code}")
    except urllib.error.URLError as exc:
        print(f"ntfy network error: {exc.reason}")
    except Exception:
        pass

def notify_test_start(msg="GeoQueryPose test started", **kwargs):
    push_ntfy(
        msg,
        title="Test Started",
        tags=("test", "start"),
        priority="default",
        **kwargs,
    )


def notify_test_end(msg="GeoQueryPose test finished", success=True, **kwargs):
    push_ntfy(
        msg,
        title="Test Finished" if success else "Test Failed",
        tags=("test", "done" if success else "failed"),
        priority="default" if success else "high",
        **kwargs,
    )
