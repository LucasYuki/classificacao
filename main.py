# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:26:35 2020

@author: lucas
"""

from tensorflow import random
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.metrics import sparse_categorical_accuracy, \
                                     categorical_crossentropy
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
                                       ModelCheckpoint

import pandas as pd
import numpy as np
import json
import sys 

import functions
from Telegram.SendTelegram import SendTelegram, send_df_as_img
from Telegram.plot_train_hist import plot_train_hist

K.set_floatx('float64')

def main():
    experiment = "Teste 1"
    
    train_len = 1000
    val_len   = 1000
    test_len  = 1000
    
    times = 10
    n_hidden = 5
    n_neurons = 256
    
    dropout_prob = 0.4
    noise = 0.1
    patience = 250
    epochs = 10000
    batch = 1000
    
    n_predictions = 100
    seed = 42
    
    activation = "tanh"
    
    config = locals()
    
    if not functions.verify_directory(experiment):
        sys.exit()
                
    # salva as configurações utilizadas no modelo
    with open(experiment + "/config.json", "w") as File:
        json.dump(config, File)
    
    index_col = ["Grandeza", "Condicao"]
    models_col  = [str(i) for i in range(times)]
    
    
    for model_num in [1, 2, 3]:
        files, columns = functions.get_files(model_num)
        
        acc_df = None
        loss_df = None
        
        #acc_df = pd.read_csv(experiment+"/Model"+str(model_num)+"/accuracy.csv",
        #                     index_col=[0, 1])
        #loss_df = pd.read_csv(experiment+"/Model"+str(model_num)+"/loss.csv",
        #                     index_col=[0, 1])
    
        for grand in columns["Grandeza"]:
            np.random.seed(seed)
            random.set_seed(seed)
            
            # prepara os dados para o treinamento
            Train, Val, Test, n_outputs = functions.get_splited_data(grand, train_len, 
                                                                     val_len, test_len,
                                                                     model_num)
        
            temp_loss = None
            temp_acc = None
            for t in range(times):
                dir_path = experiment+"/Model"+str(model_num)+"/"+grand+"/"+str(t)
            
                # verifica se a pasta já existe, se não existir cria uma nova pasta
                if not functions.verify_directory(dir_path):
                    sys.exit()
                
                # constroí o modelo
                Model_Input  = Input(shape=Train["x"].shape[1], name="input")
                hidden_layer = GaussianNoise(noise)(Model_Input)
                for i in range(n_hidden):
                    hidden_layer = Dense(n_neurons, activation=activation)(hidden_layer)
                    hidden_layer = Dropout(dropout_prob)(hidden_layer, training=True)
                Model_Output = Dense(n_outputs, activation="softmax")(hidden_layer)
            
                model = Model(Model_Input, Model_Output)
                
                model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy",
                              metrics = ["accuracy"])
                plot_model(model, dir_path + '/plot_model.png', show_shapes=True)
                
                print(model.summary())
                with open(dir_path + '\\model_description.txt', 'w') as txt:
                    model.summary(print_fn=lambda x: txt.write(x + '\n'))
                
                print("\n------------------------------------------------------------\n")
                print('The model will be save in the file ' + dir_path +"\\model.h5")
                
                callbacks=[]
                # durante o treino salva o histórico do loss e acerto
                cl = CSVLogger(dir_path+'\\train_history.csv', separator=',', append=False)
                callbacks.append(cl)
                # durante o treino salva o modelo com o melhor desempenho 
                mc = ModelCheckpoint(dir_path+"\\model.h5", monitor='val_loss', 
                                     mode='min', verbose=1, save_best_only=True)
                callbacks.append(mc)
                # se passar patience epochs sem ter melhoras o treino é encerrado
                if patience>=1: 
                    es = EarlyStopping(monitor='val_loss', mode='min', 
                                       verbose=1, patience=patience)
                    callbacks.append(es)
            
                # treina o modelo
                model.fit(Train["x"], Train["y"], epochs=epochs, 
                          validation_data= (Val["x"], Val["y"]), 
                          verbose=2, callbacks=callbacks, batch_size=batch)
                
                # envia os resutados para a conversa do telegram configurada
                SendTelegram('/message', 'model in '+dir_path+' train finished')
                plot_train_hist(dir_path)
                
                # faz a avaliação do modelo com o conjunto de teste
                acc = {}
                loss = {}
                confusion = {}
                for cond in Test["y"].keys():
                    pred = functions.predict(model, Test["x"][cond], 
                                             n_predictions=n_predictions)
                    mean = np.mean(pred, axis=1)            
                        
                    temp = np.argmax(mean, axis=1)
                    confusion[cond] = []
                    for i in range(mean.shape[1]):
                        confusion[cond].append(np.sum(temp==i))
                    
                    acc[cond] = confusion[cond][Test["y"][cond]]/mean.shape[0]
                    loss[cond] = np.mean(categorical_crossentropy(Test["y one_hot"][cond], mean))
                functions.heat_map((5, 4), list(confusion.values()), 
                                   [col+"_true" for col in Test["y"].keys()], 
                                   list(Test["y"].keys()), dir_path, 
                                   "confusion_matrix", "Confusion matrix",
                                   vmax = mean.shape[0], format_text="%.0f")
                acc["All"] = np.mean(list(acc.values()))
                loss["All"] = np.mean(list(loss.values()))
                
                index = pd.MultiIndex.from_product([[grand],list(acc.keys())],
                                                   names=index_col)
                if temp_acc is None:
                    temp_acc=pd.DataFrame(list(acc.values()), 
                                          index=index, columns=[str(t)])
                    temp_loss=pd.DataFrame(list(loss.values()), 
                                           index=index, columns=[str(t)])
                else:
                    temp_acc=temp_acc.join(pd.DataFrame(list(acc.values()),
                                                        index=index, columns=[str(t)]))
                    temp_loss=temp_loss.join(pd.DataFrame(list(loss.values()),
                                                          index=index, columns=[str(t)]))
        
                print("accuracy", acc["All"], "\nloss:", loss["All"])
                with open(dir_path+"\\acc.json", 'w') as File:
                    json.dump(acc, File)
                with open(dir_path+"\\loss.json", 'w') as File:
                    json.dump(loss, File)
                    
                try:
                    SendTelegram('/message', 'test finished')
                    SendTelegram('/message', 'Accuracy '+str(acc["All"]))
                    SendTelegram('/message', 'Loss '+str(loss["All"]))
                except:
                    print("send telegram failed")
                    
                K.clear_session()
            
            if acc_df is None:
                acc_df  = temp_acc
                loss_df = temp_loss
            else:
                acc_df  = acc_df.append(temp_acc)
                loss_df = loss_df.append(temp_loss)
            
            try:
                send_df_as_img(acc_df, "Accuracy")
                send_df_as_img(loss_df, "Loss")
            except:
                print("send telegram failed")
            
            acc_df.to_csv(experiment+"/Model"+str(model_num)+"/accuracy.csv")
            loss_df.to_csv(experiment+"/Model"+str(model_num)+"/loss.csv")

main()