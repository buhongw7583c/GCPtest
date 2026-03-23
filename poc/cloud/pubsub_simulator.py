"""
cloud/pubsub_simulator.py

Simulates Google Cloud Pub/Sub for event-driven data ingestion from edge nodes.
In production this would be the actual Cloud Pub/Sub client.

Messages published here are consumed by downstream processors (e.g. Dataflow).
"""

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional


@dataclass
class PubSubMessage:
    """A single Pub/Sub message."""
    message_id: str
    factory_id: str
    event_type: str           # "data_upload" | "model_update" | "alert"
    payload: Dict[str, Any]
    publish_time: float = field(default_factory=time.time)
    ack: bool = False

    def acknowledge(self) -> None:
        self.ack = True


class PubSubTopic:
    """Simulates a single Pub/Sub topic."""

    def __init__(self, topic_name: str, max_size: int = 1000):
        self.topic_name = topic_name
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._subscribers: List[Callable[[PubSubMessage], None]] = []
        self._message_count = 0
        self._lock = threading.Lock()

    def publish(self, factory_id: str, event_type: str, payload: Dict[str, Any]) -> str:
        """Publish a message. Returns message_id."""
        msg = PubSubMessage(
            message_id=str(uuid.uuid4())[:8],
            factory_id=factory_id,
            event_type=event_type,
            payload=payload,
        )
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            # Drop oldest message (simulate back-pressure handling)
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(msg)
        with self._lock:
            self._message_count += 1
        return msg.message_id

    def pull(self, max_messages: int = 10) -> List[PubSubMessage]:
        """Pull up to max_messages from the topic."""
        messages = []
        for _ in range(max_messages):
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def size(self) -> int:
        return self._queue.qsize()

    @property
    def total_published(self) -> int:
        with self._lock:
            return self._message_count


class PubSubSimulator:
    """
    Manages multiple Pub/Sub topics.

    Topics used in this POC:
      - "edge-data-upload"   : raw sample data from edge nodes
      - "model-updates"      : FL weight updates from edge/regional
      - "alerts"             : drift / performance alerts
      - "pipeline-triggers"  : MLOps pipeline trigger events
    """

    TOPICS = ["edge-data-upload", "model-updates", "alerts", "pipeline-triggers"]

    def __init__(self, max_queue_size: int = 1000):
        self._topics: Dict[str, PubSubTopic] = {
            name: PubSubTopic(name, max_queue_size) for name in self.TOPICS
        }

    def publish(self, topic: str, factory_id: str, event_type: str, payload: Dict[str, Any]) -> str:
        if topic not in self._topics:
            raise ValueError(f"Unknown topic: {topic}. Valid topics: {self.TOPICS}")
        return self._topics[topic].publish(factory_id, event_type, payload)

    def pull(self, topic: str, max_messages: int = 10) -> List[PubSubMessage]:
        if topic not in self._topics:
            raise ValueError(f"Unknown topic: {topic}")
        return self._topics[topic].pull(max_messages)

    def stats(self) -> Dict[str, Any]:
        return {
            topic: {
                "queue_size": t.size(),
                "total_published": t.total_published,
            }
            for topic, t in self._topics.items()
        }
