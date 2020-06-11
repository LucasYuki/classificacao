# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:21:35 2020

@author: lucas
"""

import pandas as pd
import os

def generate_directory_df(path, csv_path):
    files = []
    for file in os.listdir(path):
        temp = file.split("_")
        if len(temp)==2:
            files.append([temp[0], temp[1][:-4], "data", path+"/"+file])
        if len(temp)==3:
            files.append([temp[0], temp[1], temp[2][:-4], path+"/"+file])
    files = pd.DataFrame(files, columns=["Grandeza","Condicao","Tipo","Diretorio"])
    files = files.set_index(["Grandeza","Condicao","Tipo"])
    files.to_csv(csv_path)
    return files

# pasta onde os arquivos estão os dados
data_dir = "Dados_Modelos/Results_Model1"

# arquivo csv de destino
save_file = "exemplo.csv"

generate_directory_df(data_dir, save_file)