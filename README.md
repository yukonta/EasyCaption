# EasyCaption
Telegram Bot for Image Captioning.

This program implements the telegram-bot which make image captioning online (make text descriptions of the image) and gives out the captions.

You have to prepare an image for description. This image you can send to the telegram-bot during user-dialog.

Before running the program the set of libraries from requirements.txt must be insalled. if you use Anaconda: from Anaconda Prompt input the command (under your shell):

conda install --yes --file requirements.txt

If you use pip (without Anaconda) run the command:

pip install -r requirements.txt

Instead of installing requirements.txt you can install the next libraries:

conda create --name EnvName python=3.7  (then you should make: conda activate EnvName)
  
conda install -c anaconda scipy==1.2.1 (!!! before installation pytorch)

conda install -c pytorch pytorch

conda install -c pytorch torchvision

conda install -c anaconda pillow

conda install -c conda-forge python-telegram-bot


Then do the following:

git clone --quiet https://github.com/yukonta/EasyCaption

In the catalog EasyCaption download the file telegram_token.py which contains the telegram-token в формате token = 'token'.

Change the current catalog to EasyCaption/ .

In the file EasyCaption/config.py (in the first string) set the actual proxy SOCKS4 for Telegram. The actual proxy address you can find in the file https://proxyscrape.com/free-proxy-list (you need SOCKS4) - download the file and choose the address.

Run main.py .

Find in Telegram the Telegram-bot EasyCaption.

Send the command /start to the Telegram-bot.

Follow the instructions in the dialog: The Telegram-bot will show 2 buttons: "LOAD PICTURE" and "I don't want to continue". Press the first button for loading picture. The second button you can press if you don't want to continue.
After you press "LOAD PICTURE" the Bot will show the message "Good! Please send me an image! Or send /cancel if you don't want to.". Then you should send a picture to the Bot (!!! Important! Please, send a picture as Photo (not as File)!!!).

Then you have to wait - about 1 minute while the captions is preparing.

Then the bot will give you several captions for the picture!!!

