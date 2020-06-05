'''
This is an example of how to send data to Slack webhooks in Python with the
requests module.
Detailed documentation of Slack Incoming Webhooks:
https://api.slack.com/incoming-webhooks
'''

import json
import requests
from time import time
import base64
import io
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
  
from PIL import Image, ImageDraw

with open("Telegram//Telegram_config.json", 'r') as File:
    config=json.load(File)

webhook_url = config['webhook_url']
chat_id = config['chat_id']
    
def SendTelegram(command, text):
    slack_data = {'update_id': -1, 
                  'message': {'message_id': -1, 
                              'date': time(), 
                              'chat': {'id': chat_id, 'type': '', 'first_name': ''}, 
                              'text': command+' '+text, 
                              'entities': [{'type': 'bot_command', 'offset': 0, 'length': len(command)}], 
                              'caption_entities': [], 
                              'photo': [], 
                              'new_chat_members': [], 
                              'new_chat_photo': [], 
                              'delete_chat_photo': False, 
                              'group_chat_created': False, 
                              'supergroup_chat_created': False, 
                              'channel_chat_created': False, 
                              'from': {'id': 0, 'first_name': '', 'is_bot': False, 'language_code': ''}}}
    slack_data['_effective_message']=slack_data["message"].copy()
    response = requests.post(
        webhook_url, data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )
    
def SendImage(path, show=False):
    if isinstance(path, str):
        with open(path, "rb") as imageFile:
            string = base64.b64encode(imageFile.read()).decode('ascii')
    elif isinstance(path, io.BytesIO):
        path.seek(0)
        string = base64.b64encode(path.read()).decode('ascii')
    
    print()
    if show:
        buf = io.BytesIO(base64.b64decode(string))
        Image.open(buf).show()
    
    command = '/send_fig'
    SendTelegram(command, string)
    
def SendPlot(data):
    accuracy = {'data':[list(data["epoch"]), list(data["accuracy"])], 
                'args':{'label':'train', 'lw':0.5}}
    accuracy_val = {'data':[list(data["epoch"]), list(data["val_accuracy"])], 
                'args':{'label':'validate', 'lw':0.5}}
    send = {"datas":[accuracy, accuracy_val], 
            'xlabel':"epochs", 
            'ylabel':"accuracy",
            'title':'accuracy',
            'legend':True,
            'grid':True,
            'figsize':(10, 6)}
    text = json.dumps(send)
    
    command = '/plot'    
    SendTelegram(command, text, webhook_url, chat_id)

def send_df_as_img(df, title = ""):
    img = Image.new('RGB', (550, 120))
    d = ImageDraw.Draw(img)
    size = d.multiline_textsize(title + "\n" + str(df))
    img = Image.new('RGB', (size[0]+30, size[1]+30), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.multiline_text((15,15), title + "\n" + str(df), fill=(0,0,0), align='center')
    
    buf = io.BytesIO()
    img.save(buf, format='png')
    
    SendImage(buf)
"""
#data = pd.read_csv('C:/Users/lucas/Documents/Rotating-Machinery-Imbalance-Fault-Diagnosis/Models/teste/0/train_history.csv')

SendImage(path, webhook_url, chat_id)
SendTelegram('/message', '64/M2d0/models/2', webhook_url, chat_id)
#SendPlot(data, webhook_url, chat_id)
"""