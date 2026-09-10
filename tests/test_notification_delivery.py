"""Test actual dispatcher code without Telegram requests or database writes."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class RetryAfter(Exception):
    retry_after = 10


class Forbidden(Exception):
    pass


class BadRequest(Exception):
    pass


spec = importlib.util.spec_from_file_location('delivery_under_test', ROOT / 'src/telegram_code/notification_delivery.py')
delivery = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    'telegram.error': types.SimpleNamespace(RetryAfter=RetryAfter, Forbidden=Forbidden, BadRequest=BadRequest),
    'telegram_code.database': types.SimpleNamespace(_get_db_connection=Mock()),
}):
    spec.loader.exec_module(delivery)


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = types.SimpleNamespace(application=types.SimpleNamespace(bot_data={}),
                                            bot=types.SimpleNamespace(send_message=AsyncMock(return_value=types.SimpleNamespace(message_id=42)),
                                                                      send_photo=AsyncMock(return_value=types.SimpleNamespace(message_id=43))))
        self.item = dict(id=1, userid=123, message='Actual radar\nRain has cleared nearby.')

    async def test_success_acknowledged(self):
        with patch.object(delivery, 'claim_notification', side_effect=[self.item, None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_message.assert_awaited_once()
            self.assertEqual(finish.call_args.args[1], 'sent')
            self.assertEqual(finish.call_args.args[3], 42)

    async def test_photo_uses_persisted_image_and_caption(self):
        self.item['photo'] = b'original-image'
        received = []
        async def send_photo(**kwargs):
            received.append((kwargs['photo'].read(), kwargs['caption']))
            return types.SimpleNamespace(message_id=43)
        self.context.bot.send_photo.side_effect = send_photo
        with patch.object(delivery, 'claim_notification', side_effect=[self.item, None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.assertEqual(received, [(b'original-image', self.item['message'])])
            self.context.bot.send_message.assert_not_awaited()
            self.assertEqual(finish.call_args.args[3], 43)

    async def test_photo_ack_failure_does_not_resend(self):
        self.item['photo'] = b'original-image'
        with patch.object(delivery, 'claim_notification', side_effect=[self.item, None]), patch.object(delivery, 'finish_notification', side_effect=[RuntimeError('offline'), None]):
            with self.assertRaises(RuntimeError):
                await delivery.deliver_notifications(self.context)
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_photo.assert_awaited_once()

    async def test_ack_failure_retries_database_not_telegram(self):
        with patch.object(delivery, 'claim_notification', side_effect=[self.item, None]), patch.object(delivery, 'finish_notification', side_effect=[RuntimeError('db down'), None]) as finish:
            with self.assertRaises(RuntimeError):
                await delivery.deliver_notifications(self.context)
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_message.assert_awaited_once()
            self.assertEqual(finish.call_count, 2)

    async def test_ambiguous_timeout_is_not_blindly_retried(self):
        self.context.bot.send_message.side_effect = TimeoutError()
        with patch.object(delivery, 'claim_notification', side_effect=[self.item, None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.assertEqual(finish.call_args.args[1], 'uncertain')
            self.context.bot.send_message.assert_awaited_once()

    async def test_rate_limit_goes_back_to_pending(self):
        self.context.bot.send_message.side_effect = RetryAfter()
        with patch.object(delivery, 'claim_notification', side_effect=[self.item, None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.assertEqual(finish.call_args.args[1], 'pending')
            self.assertEqual(finish.call_args.args[4], 10)

    async def test_cancelled_setting_or_location_never_sends(self):
        with patch.object(delivery, 'claim_notification', side_effect=[{'cancelled': True}, None]):
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_message.assert_not_awaited()


class ClaimTests(unittest.TestCase):
    def claim(self, mode='automatic', version=1, actual=False, expired=False):
        from datetime import datetime, timedelta
        now = datetime(2026,9,10,12)
        item = dict(id=1,userid=123,settings_version=1,reason=delivery.ENDED if actual else 'Rain is predicted at your location',
                    forecast_at=now + timedelta(minutes=-1 if expired else 5))
        cursor = Mock()
        cursor.fetchone.side_effect = [item, {'mode':mode,'rain_settings_version':version}]
        connection = Mock()
        connection.cursor.return_value = cursor
        with patch.object(delivery, '_get_db_connection', return_value=connection):
            result = delivery.claim_notification(now)
        connection.commit.assert_called_once()
        return result

    def test_manual_mode_cancels(self):
        self.assertEqual(self.claim(mode='manual'), {'cancelled':True})

    def test_changed_location_or_settings_cancels(self):
        self.assertEqual(self.claim(version=2), {'cancelled':True})

    def test_expired_forecast_cancelled_but_actual_end_delivers(self):
        self.assertEqual(self.claim(expired=True), {'cancelled':True})
        self.assertEqual(self.claim(actual=True,expired=True)['id'],1)


if __name__ == '__main__':
    unittest.main()
