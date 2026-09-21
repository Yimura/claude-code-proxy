"""Ownership primitives for pre-subscribed control response streams."""

from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from ..event_journal import Subscription


class SubscriptionOwner:
    def __init__(self, subscription: Subscription) -> None:
        self._subscription: Subscription | None = subscription
        self._closed = False

    @property
    def subscription(self) -> Subscription:
        if self._subscription is None:
            raise RuntimeError("performance subscription is not available")
        return self._subscription

    def release(self, original: BaseException | None = None) -> None:
        subscription = self._subscription
        self._subscription = None
        if subscription is None:
            return
        try:
            subscription.close()
        except BaseException:
            if original is None:
                raise

    def replace(self, subscription: Subscription) -> None:
        if self._closed or self._subscription is not None:
            subscription.close()
            raise RuntimeError("performance subscription owner is closed")
        self._subscription = subscription

    def close(self, original: BaseException | None = None) -> None:
        self._closed = True
        self.release(original)


class OwnedPerformanceStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncIterator[str],
        owner: SubscriptionOwner,
    ) -> None:
        super().__init__(content, media_type="application/x-ndjson")
        self._owner = owner

    async def __call__(self, scope, receive, send) -> None:
        original: BaseException | None = None
        try:
            await super().__call__(scope, receive, send)
        except BaseException as error:
            original = error
            raise
        finally:
            self._owner.close(original)
