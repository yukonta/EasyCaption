from model import ImageCaptioningModel

from telegram_token import token
from config import ProxyURL, StartMsg, WantTalkMsg1, CancelMsg, WaitDescriptionMsg, NextActMsg
from io import BytesIO

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler)
import logging

MAX_CAPTION_LEN = 100
TOKENS_FILE_NAME = 'tokens_img_capt.txt'
"""
Send /start to initiate the conversation.
"""

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

model = ImageCaptioningModel()
first_image_file = {}

PICT_FOR_DESCR, NEXT_PHOTO, WANT_TALK, NEXT_ACT = range(4)
reply_keyboard = [['LOAD PICTURE', 'I do not want to continue']]


# реакция на "/start"
def start(update, context):
    print('User Start')
    update.message.reply_text(
        StartMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))

    return WANT_TALK


# метод  - реакция на нажатие кнопки LOAD PICTURE
def want_talk(update, context):
    text = update.message.text
    print(text)
    if text == reply_keyboard[0][0]:
        update.message.reply_text(WantTalkMsg1, reply_markup=ReplyKeyboardRemove())
        return PICT_FOR_DESCR
    else:
        return cancel(update, context)

# реакция на "/cancel"
def cancel(update, context):
    print('User Cancel')
    update.message.reply_text(CancelMsg, reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

# Получает картинку от пользователя (из Телеграм-Бота) и отправляет ее в функцию image_captioning. Выдает в Телеграм-Бот пользователю сгенерированные out_captions
def send_pict_for_descr(update, context):
    update.message.reply_text(WaitDescriptionMsg)
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = image_info.get_file()
    print('    -the image to describe')
    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)

    out_captions = model.image_captioning(content_image_stream)
    # теперь отправим назад out_captions
    update.message.reply_text(out_captions)
    update.message.reply_text(
        NextActMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    print("Sent out_captions")
    return WANT_TALK


if __name__ == '__main__':
    # используем прокси"socks4 proxy"

    # создаём апдейтер и передаём им наш токен, который был выдан после создания бота
    updater = Updater(token=token, use_context=True, request_kwargs={'proxy_url': ProxyURL})
    # определяем диспетчер для регистрации обработчиков
    dp = updater.dispatcher
    # инициируем обработчики для диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            WANT_TALK: [MessageHandler(Filters.regex('^(LOAD PICTURE|I do not want to continue)$'), want_talk)],
            PICT_FOR_DESCR: [MessageHandler(Filters.photo, send_pict_for_descr)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    # Останавливаем бота, если были нажаты Ctrl + C
    updater.idle()
