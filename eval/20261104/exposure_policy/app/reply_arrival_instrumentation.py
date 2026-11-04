"""Run-scoped, opt-in observation of Flower GrpcGrid reply polls.

The original Grid methods are called exactly once and their objects are returned
unchanged. Times are server monotonic seconds; reply visibility is bounded by
the pull call, not the network arrival instant.
"""
import json
import time
from collections.abc import Collection
from pathlib import Path


def install(grid, path, experiment_id):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_send = grid.send_and_receive
    original_push = grid.push_messages
    original_pull = grid.pull_messages
    state = {"call": 0, "active": None, "first": set()}

    def emit(event, **fields):
        row = {"event": event, "experiment_id": experiment_id,
               "server_monotonic_s": time.monotonic(),
               "server_unix_s": time.time(), "call": state["active"], **fields}
        with path.open("a", buffering=1) as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def push(messages):
        # The strategy passes a list. Do not consume a one-shot iterable.
        descriptors = []
        if isinstance(messages, (list, tuple)):
            for msg in messages:
                meta = msg.metadata
                descriptors.append({"dst_node_id": int(meta.dst_node_id),
                                    "group_id": str(meta.group_id),
                                    "message_type": str(meta.message_type)})
        emit("push_start", request_count=len(descriptors) if descriptors else None)
        ids = original_push(messages)
        emit("push_end", request_ids=list(ids) if isinstance(ids, Collection) else None,
             requests=descriptors)
        return ids

    def pull(ids):
        requested = list(ids) if isinstance(ids, Collection) else None
        start = time.monotonic()
        emit("pull_start", outstanding_request_ids=requested)
        replies = original_pull(ids)
        end = time.monotonic()
        observed = []
        for reply in replies:
            meta = reply.metadata
            request_id = str(meta.reply_to_message_id)
            item = {"request_id": request_id, "response_id": str(meta.message_id),
                    "src_node_id": int(meta.src_node_id)}
            observed.append(item)
            if request_id not in state["first"]:
                state["first"].add(request_id)
                emit("first_reply_observed", **{
                     **item, "poll_start_monotonic_s": start,
                     "poll_end_monotonic_s": end})
        emit("pull_end", poll_start_monotonic_s=start,
             poll_end_monotonic_s=end, observed_replies=observed)
        return replies

    def send(messages, timeout=None):
        state["call"] += 1
        state["active"] = state["call"]
        state["first"] = set()
        emit("send_and_receive_start", timeout=timeout)
        try:
            result = original_send(messages, timeout=timeout)
            emit("all_replies_returned", reply_count=len(result))
            return result
        finally:
            state["active"] = None

    grid.push_messages = push
    grid.pull_messages = pull
    grid.send_and_receive = send
    emit("instrumentation_enabled", flower_grid_type=type(grid).__name__)
    return emit
