# -*- coding: utf-8 -*-
"""
@author: Carlos
"""

import pandas as pd

import FuncionesP2 as p2

import os
import errno

for i in range(3):
    try:
        os.mkdir('Caso{}'.format(str(i+1)))
        os.mkdir('Caso{}/X'.format(str(i+1)))
        os.mkdir('Caso{}/Y'.format(str(i+1)))
        os.mkdir('Caso{}/X/kmeans'.format(str(i+1)))
        os.mkdir('Caso{}/X/meanshift'.format(str(i+1)))
        os.mkdir('Caso{}/X/dbscan'.format(str(i+1)))
        os.mkdir('Caso{}/X/jerarquico'.format(str(i+1)))
        os.mkdir('Caso{}/X/birch'.format(str(i+1)))
        os.mkdir('Caso{}/Y/kmeans'.format(str(i+1)))
        os.mkdir('Caso{}/Y/meanshift'.format(str(i+1)))
        os.mkdir('Caso{}/Y/dbscan'.format(str(i+1)))
        os.mkdir('Caso{}/Y/jerarquico'.format(str(i+1)))
        os.mkdir('Caso{}/Y/birch'.format(str(i+1)))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

SEED = 1092000
def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

datos = pd.read_csv("./Fire-Incidents_v2.csv")



"""
Caso 1
"""

caso1 = datos
caso1_c = datos
caso1 = caso1[caso1["Method_Of_Fire_Control"] == "Extinguished by fire department"]
caso1_c = caso1_c[caso1_c["Method_Of_Fire_Control"] != "Extinguished by fire department"]
caso1 = caso1[caso1.Month.isin(["June", "July", "August", "September"])]
caso1_c = caso1_c[~caso1_c.Month.isin(["June", "July", "August", "September"])]
caso1 = caso1.rename(columns={"Civilian_Casualties": "Muertos", "Count_of_Persons_Rescued": "Rescatados", "Estimated_Dollar_Loss": "Coste", "Estimated_Number_Of_Persons_Displaced": "Desplazados", "Arrival_Time": "Tiempo"})
caso1_c = caso1_c.rename(columns={"Civilian_Casualties": "Muertos", "Count_of_Persons_Rescued": "Rescatados", "Estimated_Dollar_Loss": "Coste", "Estimated_Number_Of_Persons_Displaced": "Desplazados", "Arrival_Time": "Tiempo"})

usadas1 = ["Muertos", "Rescatados", "Coste", "Desplazados", "Tiempo"]

print("El caso de estudio tiene {} tuplas".format(str(caso1[usadas1].size)), end='\n')
print(caso1[usadas1].isna().sum(), end='\n\n')
print(caso1[usadas1].astype(bool).sum(axis=0), end='\n\n')

"""
for col in usadas1:
   caso1[col] = caso1[col].fillna(0, inplace=True)

print(caso1[usadas1].astype(bool).sum(axis=0), end="\n")
"""


probar1 = True
if (probar1):
    X = caso1[usadas1]
    X_norm = X.apply(norm_to_zero_one)
    Y = caso1_c[usadas1]
    Y_norm = Y.apply(norm_to_zero_one)
    
    # kmeans
    p2.kmeans_compare(X, X_norm, 11, 5, SEED, usadas1, "Caso1/X")
    p2.kmeans(X, X_norm, 2, 5, SEED, usadas1, "Caso1/X")
    p2.kmeans_compare(Y, Y_norm, 11, 5, SEED, usadas1, "Caso1/Y")
    p2.kmeans(Y, Y_norm, 2, 5, SEED, usadas1, "Caso1/Y")
    
    # meanshift
    p2.meanshift(X, X_norm, SEED, usadas1, "Caso1/X")
    p2.meanshift(Y, Y_norm, SEED, usadas1, "Caso1/Y")
    
    # dbscan
    p2.dbscan_compare(X, X_norm, 0.4, SEED, usadas1, "Caso1/X")
    p2.dbscan(X, X_norm, 0.3, SEED, usadas1, "Caso1/X")
    p2.dbscan_compare(Y, Y_norm, 0.4, SEED, usadas1, "Caso1/Y")
    p2.dbscan(Y, Y_norm, 0.3, SEED, usadas1, "Caso1/Y")
    
    # jerárquico
    p2.jerarquico(X, 2, SEED, usadas1, "Caso1/X")
    p2.jerarquico(Y, 2, SEED, usadas1, "Caso1/Y")
    
    # birch
    p2.birch(X, X_norm, 2, 0.1, SEED, usadas1, "Caso1/X")
    p2.birch(Y, Y_norm, 2, 0.1, SEED, usadas1, "Caso1/Y")


"""
Caso 2
"""

caso2 = datos
caso2_c = datos
caso2 = caso2[caso2["Civilian_Casualties"] > 0]
#caso2_c = caso2_c[caso2_c["Civilian_Casualties"] > 0
caso2 = caso2[caso2["Fire_Alarm_System_Operation"] == "Fire alarm system operated"]
caso2_c = caso2_c[caso2_c["Fire_Alarm_System_Operation"] != "Fire alarm system operated"]
caso2 = caso2[caso2.Fire_Alarm_System_Impact_on_Evacuation.isin(["Some persons (at risk) evacuated as a result of hearing fire alarm system",
                                                                 "All persons (at risk of injury) evacuated as a result of hearing fire alarm system"])]
caso2_c = caso2_c[~caso2_c.Fire_Alarm_System_Impact_on_Evacuation.isin(["Some persons (at risk) evacuated as a result of hearing fire alarm system",
                                                                 "All persons (at risk of injury) evacuated as a result of hearing fire alarm system"])]
caso2 = caso2.rename(columns={"Civilian_Casualties": "Muertos", "Count_of_Persons_Rescued": "Rescatados", "Estimated_Dollar_Loss": "Coste", "Estimated_Number_Of_Persons_Displaced": "Desplazados", "Arrival_Time": "Tiempo"})
caso2_c = caso2_c.rename(columns={"Civilian_Casualties": "Muertos", "Count_of_Persons_Rescued": "Rescatados", "Estimated_Dollar_Loss": "Coste", "Estimated_Number_Of_Persons_Displaced": "Desplazados", "Arrival_Time": "Tiempo"})

usadas2 = ["Muertos", "Rescatados", "Coste", "Desplazados", "Tiempo"]

print("El caso de estudio tiene {} tuplas".format(str(caso2[usadas2].size)), end='\n')
print(caso2[usadas2].isna().sum(), end='\n\n')
print(caso2[usadas2].astype(bool).sum(axis=0), end='\n\n')


probar2 = True
if (probar2):
    X = caso2[usadas2]
    X_norm = X.apply(norm_to_zero_one)
    Y = caso2_c[usadas2]
    Y_norm = Y.apply(norm_to_zero_one)

    # kmeans
    p2.kmeans_compare(X, X_norm, 11, 5, SEED, usadas2, "Caso2/X")
    p2.kmeans(X, X_norm, 3, 5, SEED, usadas2, "Caso2/X")
    p2.kmeans_compare(Y, Y_norm, 11, 5, SEED, usadas2, "Caso2/Y")
    p2.kmeans(Y, Y_norm, 3, 5, SEED, usadas2, "Caso2/Y")
    
    # meanshift
    p2.meanshift(X, X_norm, SEED, usadas2, "Caso2/X")
    p2.meanshift(Y, Y_norm, SEED, usadas2, "Caso2/Y")
    
    # dbscan
    p2.dbscan_compare(X, X_norm, 0.6, SEED, usadas2, "Caso2/X")
    p2.dbscan(X, X_norm, 0.55, SEED, usadas2, "Caso2/X")
    p2.dbscan_compare(Y, Y_norm, 0.6, SEED, usadas2, "Caso2/Y")
    p2.dbscan(Y, Y_norm, 0.55, SEED, usadas2, "Caso2/Y")
    
    # jerárquico
    p2.jerarquico(X, 3, SEED, usadas2, "Caso2/X")
    p2.jerarquico(Y, 3, SEED, usadas2, "Caso2/Y")
    
    # birch
    p2.birch(X, X_norm, 3, 0.2, SEED, usadas2, "Caso2/X")
    p2.birch(Y, Y_norm, 3, 0.2, SEED, usadas2, "Caso2/Y")


"""
Caso 3
"""

caso3 = datos
caso3_c = datos
caso3 = caso3[caso3["Ignition_Source"] == "Smoker's Articles (eg. cigarettes, cigars, pipes already ignited"]
caso3_c = caso3_c[caso3_c["Ignition_Source"] != "Smoker's Articles (eg. cigarettes, cigars, pipes already ignited"]
caso3 =  caso3[caso3.Day_Of_Week.isin(["Friday", "Saturday", "Sunday"])]
caso3_c =  caso3_c[~caso3_c.Day_Of_Week.isin(["Friday", "Saturday", "Sunday"])]
caso3 = caso3[caso3.Month.isin(["June", "July", "August", "September"])]
caso3_c = caso3_c[~caso3_c.Month.isin(["June", "July", "August", "September"])]
caso3 = caso3.rename(columns={"Civilian_Casualties": "Muertos", "Count_of_Persons_Rescued": "Rescatados", "Estimated_Dollar_Loss": "Coste", "Estimated_Number_Of_Persons_Displaced": "Desplazados", "Arrival_Time": "Tiempo"})
caso3_c= caso3_c.rename(columns={"Civilian_Casualties": "Muertos", "Count_of_Persons_Rescued": "Rescatados", "Estimated_Dollar_Loss": "Coste", "Estimated_Number_Of_Persons_Displaced": "Desplazados", "Arrival_Time": "Tiempo"})

usadas3 = ["Muertos", "Rescatados", "Coste", "Desplazados", "Tiempo"]

print("El caso de estudio tiene {} tuplas".format(str(caso3[usadas3].size)), end='\n')
print(caso3[usadas3].isna().sum(), end='\n\n')
print(caso3[usadas3].astype(bool).sum(axis=0), end='\n\n')

probar3 = True
if (probar3):
    X = caso3[usadas3]
    X_norm = X.apply(norm_to_zero_one)
    Y = caso3_c[usadas3]
    Y_norm = Y.apply(norm_to_zero_one)

    # kmeans
    p2.kmeans_compare(X, X_norm, 11, 5, SEED, usadas3, "Caso3/X")
    p2.kmeans(X, X_norm, 4, 5, SEED, usadas3, "Caso3/X")
    p2.kmeans_compare(Y, Y_norm, 11, 5, SEED, usadas3, "Caso3/Y")
    p2.kmeans(Y, Y_norm, 4, 5, SEED, usadas3, "Caso3/Y")

    
    # meanshift
    p2.meanshift(X, X_norm, SEED, usadas3, "Caso3/X")
    p2.meanshift(Y, Y_norm, SEED, usadas3, "Caso3/Y")
    
    # dbscan
    p2.dbscan_compare(X, X_norm, 0.4, SEED, usadas3, "Caso3/X")
    p2.dbscan(X, X_norm, 0.35, SEED, usadas3, "Caso3/X")
    p2.dbscan_compare(Y, Y_norm, 0.4, SEED, usadas3, "Caso3/Y")
    p2.dbscan(Y, Y_norm, 0.35, SEED, usadas3, "Caso3/Y")
    
    # jerárquico
    p2.jerarquico(X, 4, SEED, usadas3, "Caso3/X")
    p2.jerarquico(Y, 4, SEED, usadas3, "Caso3/Y")
    
    # birch
    p2.birch(X, X_norm, 4, 0.2, SEED, usadas3, "Caso3/X")
    p2.birch(Y, Y_norm, 4, 0.2, SEED, usadas3, "Caso3/Y")