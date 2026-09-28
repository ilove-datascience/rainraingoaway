"""Test actual dispatcher code without Telegram requests or database writes."""
import asyncio
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
# Load native dependencies before patch.dict restores sys.modules after the DB stub.
from telegram_code.notification_text import encode_alert
from telegram_code import daily_forecast


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
        weather = patch.object(delivery, 'get_daily_forecast', AsyncMock(return_value=None))
        weather.start()
        self.addCleanup(weather.stop)
        recovery = patch.object(delivery, 'recover_abandoned_notifications')
        recovery.start()
        self.addCleanup(recovery.stop)
        self.context = types.SimpleNamespace(application=types.SimpleNamespace(bot_data={}),
                                            bot=types.SimpleNamespace(send_message=AsyncMock(return_value=types.SimpleNamespace(message_id=42)),
                                                                      send_photo=AsyncMock(return_value=types.SimpleNamespace(message_id=43))))
        self.item = dict(id=1, userid=123, location_id=0,
                         reason=delivery.OBSERVED, observed_at=datetime(2026,9,28,12),
                         forecast_at=datetime(2026,9,28,12,10))
        from telegram_code.notification_text import encode_alert
        self.item['message'] = encode_alert('Main', delivery.OBSERVED, 'Light')

    async def test_success_acknowledged(self):
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_message.assert_awaited_once()
            self.assertEqual(finish.call_args.args[1], 'sent')
            self.assertEqual(finish.call_args.args[3], 42)

    async def test_alert_menu_is_selected_for_destination_chat(self):
        self.item['userid'] = -1001234567890
        menu = Mock(return_value='group-menu')
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None]), patch.object(delivery, 'finish_notification'):
            await delivery.deliver_notifications(self.context, reply_markup=menu)
        menu.assert_called_once_with(-1001234567890)
        self.assertEqual(self.context.bot.send_message.await_args.kwargs['reply_markup'], 'group-menu')

    async def test_group_alert_targets_group_chat(self):
        self.item['userid'] = -1001234567890
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None]), patch.object(delivery, 'finish_notification'):
            await delivery.deliver_notifications(self.context)
        self.assertEqual(self.context.bot.send_message.await_args.kwargs['chat_id'], -1001234567890)

    async def test_photo_uses_persisted_image_and_caption(self):
        self.item['photo'] = b'original-image'
        received = []
        async def send_photo(**kwargs):
            received.append((kwargs['photo'].read(), kwargs['caption']))
            return types.SimpleNamespace(message_id=43)
        self.context.bot.send_photo.side_effect = send_photo
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.assertEqual(received, [(b'original-image', delivery.compose_notice([self.item]))])
            self.context.bot.send_message.assert_not_awaited()
            self.assertEqual(finish.call_args.args[3], 43)

    async def test_photo_ack_failure_does_not_resend(self):
        self.item['photo'] = b'original-image'
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None, None]), patch.object(delivery, 'finish_notification', side_effect=[RuntimeError('offline'), None]):
            await delivery.deliver_notifications(self.context)
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_photo.assert_awaited_once()

    async def test_ack_failure_retries_database_not_telegram(self):
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None, None]), patch.object(delivery, 'finish_notification', side_effect=[RuntimeError('db down'), None]) as finish:
            await delivery.deliver_notifications(self.context)
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_message.assert_awaited_once()
            self.assertEqual(finish.call_count, 2)

    async def test_ambiguous_timeout_is_not_blindly_retried(self):
        self.context.bot.send_message.side_effect = TimeoutError()
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.assertEqual(finish.call_args.args[1], 'uncertain')
            self.context.bot.send_message.assert_awaited_once()

    async def test_rate_limit_goes_back_to_pending(self):
        self.context.bot.send_message.side_effect = RetryAfter()
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
            self.assertEqual(finish.call_args.args[1], 'pending')
            self.assertEqual(finish.call_args.args[4], 10)

    async def test_cancelled_setting_or_location_never_sends(self):
        with patch.object(delivery, 'claim_notifications', side_effect=[[{'cancelled': True}], None]):
            await delivery.deliver_notifications(self.context)
            self.context.bot.send_message.assert_not_awaited()

    def second_item(self, **changes):
        from telegram_code.notification_text import encode_alert
        return dict(self.item, **{'id': 2, 'location_id': 7,
                    'message': encode_alert('Office', delivery.ENDED),
                    'reason': delivery.ENDED, **changes})

    async def test_two_locations_one_photo_and_both_receipts(self):
        self.item['photo'] = b'shared-map'
        second = self.second_item()
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item, second], None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
        self.context.bot.send_photo.assert_awaited_once()
        self.context.bot.send_message.assert_not_awaited()
        caption = self.context.bot.send_photo.await_args.kwargs['caption']
        self.assertIn('Main: Light rain detected', caption)
        self.assertIn('Office: Rain cleared', caption)
        self.assertEqual(caption.count('Rain update'), 1)
        self.assertEqual([call.args[0]['id'] for call in finish.call_args_list], [1, 2])
        self.assertEqual([call.args[3] for call in finish.call_args_list], [43, 43])

    async def test_mixed_maps_deliver_one_text_not_two_photos(self):
        self.item['photo'] = b'first-map'
        second = self.second_item(photo=b'other-map')
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item, second], None]), patch.object(delivery, 'finish_notification'):
            await delivery.deliver_notifications(self.context)
        self.context.bot.send_message.assert_awaited_once()
        self.context.bot.send_photo.assert_not_awaited()

    async def test_partial_batch_ack_failure_never_resends_either_location(self):
        second = self.second_item()
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item, second], None, None]), patch.object(delivery, 'finish_notification', side_effect=[RuntimeError('offline'), None, None]) as finish:
            await delivery.deliver_notifications(self.context)
            self.assertEqual(set(self.context.application.bot_data['notification_receipts']), {1})
            await delivery.deliver_notifications(self.context)
        self.context.bot.send_message.assert_awaited_once()
        self.assertEqual([call.args[0]['id'] for call in finish.call_args_list], [1, 2, 1])

    async def test_uncertain_batch_is_not_split_or_resent(self):
        self.context.bot.send_message.side_effect = TimeoutError()
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item, self.second_item()], None]), patch.object(delivery, 'finish_notification') as finish:
            await delivery.deliver_notifications(self.context)
        self.context.bot.send_message.assert_awaited_once()
        self.assertEqual([call.args[1] for call in finish.call_args_list], ['uncertain', 'uncertain'])

    async def test_six_long_labels_remain_one_complete_text_message(self):
        from telegram_code.notification_text import encode_alert
        items = [dict(self.item, id=i+1, location_id=i,
                      message=encode_alert(str(i) + '🌧' * 79, delivery.ENDED),
                      reason=delivery.ENDED, photo=b'shared-map') for i in range(6)]
        with patch.object(delivery, 'claim_notifications', side_effect=[items, None]), patch.object(delivery, 'finish_notification'):
            await delivery.deliver_notifications(self.context)
        self.context.bot.send_photo.assert_not_awaited()
        self.context.bot.send_message.assert_awaited_once()
        text = self.context.bot.send_message.await_args.kwargs['text']
        self.assertEqual(text.count('Rain cleared'), 6)
        self.assertLessEqual(len(text.encode('utf-16-le')) // 2, 4096)

    async def test_outlook_only_on_first_notice_and_never_repeated_per_location(self):
        self.item['first_today'] = True
        second = self.second_item()
        later = dict(self.item, id=3, first_today=False)
        with patch.object(delivery, 'get_daily_forecast', AsyncMock(return_value={'weather': 'fixture'})), patch.object(delivery, 'format_daily_forecast', return_value='Official outlook fixture'), patch.object(delivery, 'claim_notifications', side_effect=[[self.item, second], [later], None]), patch.object(delivery, 'finish_notification'):
            await delivery.deliver_notifications(self.context)
        texts = [call.kwargs['text'] for call in self.context.bot.send_message.await_args_list]
        self.assertEqual(texts[0].count('Official outlook fixture'), 1)
        self.assertNotIn('Official outlook fixture', texts[1])

    async def test_daily_outlook_unavailable_does_not_block_rain_notice(self):
        self.item['first_today'] = True
        with patch.object(delivery, 'claim_notifications', side_effect=[[self.item], None]), patch.object(delivery, 'finish_notification'):
            await delivery.deliver_notifications(self.context)
        self.assertIn('Main: Light rain detected', self.context.bot.send_message.await_args.kwargs['text'])


class ClaimTests(unittest.TestCase):
    def claim(self, mode='automatic', version=1, actual=False, expired=False):
        from datetime import datetime, timedelta
        now = datetime(2026,9,10,12)
        item = dict(id=1,userid=123,settings_version=1,reason=delivery.ENDED if actual else 'Rain is predicted at your location',
                    observed_at=now-timedelta(minutes=10 if expired else 0),
                    forecast_at=now + timedelta(minutes=-1 if expired else 5),message='legacy')
        cursor = Mock()
        cursor.fetchone.side_effect = [{'userid':123}, None, {'mode':mode,'rain_settings_version':version,'label':'Main'}, None]
        cursor.fetchall.side_effect = [[item], [{'location_id':0,'rain_settings_version':version}]]
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

    def test_expired_forecast_and_actual_clear_are_cancelled(self):
        self.assertEqual(self.claim(expired=True), {'cancelled':True})
        self.assertEqual(self.claim(actual=True,expired=True), {'cancelled':True})
        self.assertEqual(self.claim(actual=True)['id'],1)


if __name__ == '__main__':
    unittest.main()
