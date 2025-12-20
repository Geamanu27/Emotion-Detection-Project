from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque, Optional


class Verdict(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NO_FACE = "no_face"


@dataclass
class MoodConfig:
    window_minutes: int = 10
    negative_threshold: int = 5
    break_cooldown_min: int = 15
    motivation_cooldown_min: int = 10


class MoodEngine:
    def __init__(self, cfg: MoodConfig):
        self.cfg = cfg
        self.history = deque(maxlen=cfg.window_minutes)  # type: Deque[Verdict]
        self.break_cooldown_until = None  # type: Optional[datetime]
        self.motivation_cooldown_until = None  # type: Optional[datetime]

    def push(self, verdict: Verdict) -> None:
        self.history.append(verdict)

    def negatives_in_window(self) -> int:
        return sum(1 for v in self.history if v == Verdict.NEGATIVE)

    def can_show_break(self, now: datetime) -> bool:
        if self.break_cooldown_until is not None and now < self.break_cooldown_until:
            return False
        return True

    def can_show_motivation(self, now: datetime) -> bool:
        if self.motivation_cooldown_until is not None and now < self.motivation_cooldown_until:
            return False
        return True

    def should_break_alert(self, now: datetime) -> bool:
        if len(self.history) < min(5, self.cfg.window_minutes):
            return False

        last = self.history[-1]
        negs = self.negatives_in_window()

        return (
            last == Verdict.NEGATIVE
            and negs >= self.cfg.negative_threshold
            and self.can_show_break(now)
        )

    def should_motivate(self, now: datetime) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        return last == Verdict.POSITIVE and self.can_show_motivation(now)

    def mark_break_shown(self, now: datetime) -> None:
        self.break_cooldown_until = now + timedelta(minutes=self.cfg.break_cooldown_min)

    def mark_motivation_shown(self, now: datetime) -> None:
        self.motivation_cooldown_until = now + timedelta(minutes=self.cfg.motivation_cooldown_min)
