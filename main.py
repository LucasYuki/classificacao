# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:26:35 2020

@author: lucas
"""

#pedir para o lobato salvar os novos dados no mesmo formato que está atualmente

#criar função para retomar treino
#gerar e salvar sementes para tudo

from tensorflow import random
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
                                       ModelCheckpoint

import pandas as pd
import numpy as np
import json
import sys 

import functions
from Telegram.SendTelegram import SendTelegram, send_df_as_img, SendImage

K.set_floatx('float64')

class model1:
    def __init__(self,
                 noise     = 0,  # Desvio padrão do ruído inserido na entrada.
                 n_hidden  = 1,  # Quantidade de camadas intemediárias.
                 n_neurons = 32, # Quantidade de neurônios por camada.
                 activation = "tanh", # Ativação das camadas intermediárias.
                 dropout_prob = 0): # Probabilidade de um neurônio não ser utilizado.
        
        name = "model1"
        self.name         = name
        self.noise        = noise
        self.n_hidden     = n_hidden
        self.n_neurons    = n_neurons
        self.activation   = activation
        self.dropout_prob = dropout_prob
        
        temp = locals().copy()
        del temp["self"]
        self.config = temp
    
    def get_builder(self, input_shape, n_outputs):
        def model_build(dir_path):
            Model_Input  = Input(shape=input_shape, name="input")
            hidden_layer = GaussianNoise(self.noise)(Model_Input)
            for i in range(self.n_hidden):
                hidden_layer = Dense(self.n_neurons, activation=self.activation)(hidden_layer)
                hidden_layer = Dropout(self.dropout_prob)(hidden_layer, training=True)
            Model_Output = Dense(n_outputs, activation="softmax")(hidden_layer)
        
            model = Model(Model_Input, Model_Output)
            
            model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy",
                          metrics = ["accuracy"])
            
            # salva uma esquemático do modelo e sua descrição
            plot_model(model, dir_path + '/plot_model.png', show_shapes=True)
            
            print(model.summary())
            with open(dir_path + '\\model_description.txt', 'w') as txt:
                model.summary(print_fn=lambda x: txt.write(x + '\n'))
            
            print("\n------------------------------------------------------------\n")
            print('The model will be save in the file ' + dir_path +"\\model.h5")
            return model
        return model_build
    
    def get_config(self):
        return self.config

def evaluate(Test, model, n_per_pred):
    acc = {}
    loss = {}
    confusion = {}
    for cond in Test["y"].keys():
        pred = functions.predict(model, Test["x"][cond], 
                                 n_per_pred=n_per_pred)
        mean = np.mean(pred, axis=1)            
            
        temp = np.argmax(mean, axis=1)
        confusion[cond] = []
        for i in range(mean.shape[1]):
            confusion[cond].append(np.sum(temp==i))
        
        acc[cond] = confusion[cond][Test["y"][cond]]/mean.shape[0]
        loss[cond] = np.mean(categorical_crossentropy(Test["y one_hot"][cond], mean))

    acc["All"] = np.mean(list(acc.values()))
    loss["All"] = np.mean(list(loss.values()))
    return acc, loss, confusion
    
def train_models(times, # Quantidade de redes que serão treinadas.
                 path,  # Local onde será salvas as redes.
                 Train, # Dicionário dos dados de treino.
                 Val,   # Dicionário dos dados de validação.
                 Test,  # Dicionário dos dados de teste.
                 df_index, # Index para o dataframe de acerto e loss.
                 model_builder, # Função que gera a rede neural.
                 patience = -1, # Após patience epochs sem melhoras, o treino 
                                #é interrompido. Se patience<1, não será usado.
                 epochs = 1000, # Máximo de vezes que a rede será treinada
                                #com todos os dados de treino.
                 batch = 100, # Quantidade de dados de treino que serão 
                              #processados entre cada atualização dos pesos.
                 n_per_pred = 100, # Número de vezes que o mesmo dado irá
                                   #passar pela rede para fazer uma predição.
                 Telegram = False): # Se verdadeiro irá enviar mensagens para
                                    #a conversa do telegram configurada
    
    temp_loss = None
    temp_acc = None
    for t in range(times):
        dir_path = path+str(t)
    
        # verifica se a pasta já existe, se não existir cria uma nova pasta
        if not functions.verify_directory(dir_path):
            sys.exit()
        
        callbacks=[]
        # durante o treino salva o histórico do loss e acerto
        callbacks.append(CSVLogger(dir_path+'\\train_history.csv', 
                                   separator=',', append=False))
        # durante o treino salva o modelo com o melhor desempenho 
        callbacks.append(ModelCheckpoint(dir_path+"\\model.h5", 
                                         monitor='val_loss', mode='min', 
                                         verbose=1, save_best_only=True))
        
        # se passar patience epochs sem ter melhoras o treino é encerrado
        if patience>=1: 
            es = EarlyStopping(monitor='val_loss', mode='min', 
                               verbose=1, patience=patience)
            callbacks.append(es)
    
        # constroí e treina o modelo
        model = model_builder(dir_path)
        model.fit(Train["x"], Train["y"], epochs=epochs, 
                  validation_data= (Val["x"], Val["y"]), 
                  verbose=2, callbacks=callbacks, batch_size=batch)
        
        # gera gráficos 
        hist_image = functions.plot_train_hist(dir_path) 
        
        if Telegram:
            for img in hist_image:
                SendImage(img, show=False)
            SendTelegram('/message', 'model in '+dir_path+' train finished')
        
        # faz a avaliação do modelo com o conjunto de teste
        acc, loss, confusion = evaluate(Test, model, n_per_pred)
        
        functions.heat_map((5, 4), list(confusion.values()), 
                           [col+"_true" for col in Test["y"].keys()], 
                           list(Test["y"].keys()), dir_path, 
                           "confusion_matrix", "Confusion matrix",
                           vmax = n_per_pred, format_text="%.0f")
         
        if temp_acc is None:
            temp_acc=pd.DataFrame(list(acc.values()), 
                                  index=df_index, columns=[str(t)])
            temp_loss=pd.DataFrame(list(loss.values()), 
                                   index=df_index, columns=[str(t)])
        else:
            temp_acc=temp_acc.join(pd.DataFrame(list(acc.values()),
                                                index=df_index, columns=[str(t)]))
            temp_loss=temp_loss.join(pd.DataFrame(list(loss.values()),
                                                  index=df_index, columns=[str(t)]))

        print("accuracy", acc["All"], "\nloss:", loss["All"])
        with open(dir_path+"\\acc.json", 'w') as File:
            json.dump(acc, File)
        with open(dir_path+"\\loss.json", 'w') as File:
            json.dump(loss, File)
            
        if Telegram:
            SendTelegram('/message', 'test finished')
            SendTelegram('/message', 'Accuracy '+str(acc["All"]))
            SendTelegram('/message', 'Loss '+str(loss["All"]))
            
        K.clear_session()
    return temp_loss, temp_acc

def main(model_type,
         data_files,
         experiment = "Teste 1",
         train_len  = 1000,
         val_len    = 1000,
         test_len   = 1000,
         times      = 10,
         patience   = 250,
         epochs     = 10000,
         batch      = 1000,
         n_per_pred = 100,
         seed       = 42,
         Telegram   = False):
    
    config = locals().copy()
    config["model_type"] = model_type.get_config()
    
    if not functions.verify_directory(experiment):
        sys.exit()
                
    # salva as configurações utilizadas no modelo
    with open(experiment + "/config.json", "w") as File:
        json.dump(config, File)
    
    index_col = ["Grandeza", "Condicao"]
    models_col  = [str(i) for i in range(times)]
    
    
    for data_file in data_files:
        data_file_name = ".".join(data_file.split(".")[:-1])
        files, columns = functions.get_files(data_file)
        
        acc_df = None
        loss_df = None
        
        #caso tenha que continuar o treino pela metade
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
                                                                     data_file)
            path = experiment+"/"+data_file_name+"/"+grand+"/"
            index = pd.MultiIndex.from_product([[grand],list(Test["y"].keys())+["All"]],
                                           names=index_col)
            
            model_builder = model_type.get_builder(Train["x"].shape[1], n_outputs)
            
            temp_loss, temp_acc = train_models(times, path, Train, Val, 
                                               Test, index, model_builder,
                                               patience=patience, 
                                               epochs=epochs, batch=batch, 
                                               n_per_pred=n_per_pred)
            
            if acc_df is None:
                acc_df  = temp_acc
                loss_df = temp_loss
            else:
                acc_df  = acc_df.append(temp_acc)
                loss_df = loss_df.append(temp_loss)
            
            if Telegram:
                send_df_as_img(acc_df, "Accuracy")
                send_df_as_img(loss_df, "Loss")
            
            acc_df.to_csv(experiment+"/"+data_file_name+"/accuracy.csv")
            loss_df.to_csv(experiment+"/"+data_file_name+"/loss.csv")

if __file__.split("\\")[-1] == "main.py":
    model_config = {"n_hidden":     5,
                    "n_neurons":    256,
                    "dropout_prob": 0.4,
                    "noise":        0.1,
                    "activation":   "tanh"}
    
    experiment_config = {"experiment": "Teste 2",
                         "train_len":  1000,
                         "val_len":    1000,
                         "test_len":   1000,
                         "times":      10,
                         "patience":   250,
                         "epochs":     10000,
                         "batch":      1000,
                         "n_per_pred": 100,
                         "seed":       42,
                         "Telegram":   False}
    
    model = model1(**model_config)
    
    data_files = ["arquivos_Model1.csv", 
                  "arquivos_Model2.csv",
                  "arquivos_Model3.csv"]
    
    main(model_type=model, data_files=data_files, **experiment_config)