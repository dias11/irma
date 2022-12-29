import streamlit as st
import pandas as pd 
import numpy as np

#from matplotlib import cm  #%matplotlib inline
import matplotlib.pyplot as plt


import matplotlib as mpl
import matplotlib.colors as colors
mycolor = colors.LinearSegmentedColormap.from_list('mycolor', ['#000B65','#019EE2','#E2E201', '#EA0B00'])

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging


st.title('All interpolations')
st.text("This is a web app to explore interpolations from IRMA's Data")
uploaded_file = st.file_uploader('Upload your file here')


if uploaded_file:
#    st.header('data statistics')
    df = pd.read_csv(uploaded_file)
#    st.write(df.describe())
      
    st.header('Data Header')
    st.write(df)
    
    csv_data = pd.DataFrame(df)  # read csv data into dataframe
#    st.text(csv_data)           # Print dataframe
    
    data = csv_data.to_numpy()   # Convert csv dataframe to numpy array
#    st.text(data)                #Print data numpy array

    x = data[:, 2]              # x coordinates 
#    st.text(x)

    y = data[:, 1]              # y coordinates
#    st.text(y)

    z = data[:, 5]
#    st.text(z)
    
    
#   This Part is the plotter of Weather Stations

    
    
    fig_WS,axis_WS = plt.subplots()
    axis_WS.set_title('Wheather Stations Locations')
    axis_WS.set_xlabel('x')
    axis_WS.set_ylabel('y')
    axis_WS.set_xlim(20.63, 21.15)
    axis_WS.set_ylim(39.00, 39.33)
    WSG = axis_WS.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_WS.colorbar(WSG)
    st.write(fig_WS)
    
    
#   This Part is the plotter of Kriging Exponential Interpolation    
    
    
    ok = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model = 'exponential', #exponential, gaussian, linear
    verbose = True,
    enable_plotting = True
    )
    
    gridx = np.linspace(20.63, 21.15, 100)
    gridy = np.linspace(39.00, 39.33 ,100)
    zstar, ss = ok.execute('grid', gridx, gridy )
#-----------end of calculations   
    
    fig_KR_exp,axis_KR_exp = plt.subplots()
    axis_KR_exp.set_title('Kriging Exponential')
    axis_KR_exp.set_xlabel('x')
    axis_KR_exp.set_ylabel('y')
    axis_KR_exp.set_xlim(20.63, 21.15)
    axis_KR_exp.set_ylim(39.00, 39.33)
    axis_KR_exp.contourf(gridx, gridy,  zstar,100, cmap=mycolor)
    KR_exp = axis_KR_exp.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_KR_exp.colorbar(KR_exp)
    st.write(fig_KR_exp)

    
#   This Part is the plotter of Kriging Gaussian Interpolation  

    ok = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model = 'gaussian', #exponential, gaussian, linear
    verbose = True,
    enable_plotting = True
    )
    
    gridx = np.linspace(20.63, 21.15, 100)
    gridy = np.linspace(39.00, 39.33 ,100)
    zstar, ss = ok.execute('grid', gridx, gridy )


    fig_KR_gau,axis_KR_gau = plt.subplots()
    axis_KR_gau.set_title('Kriging Gaussian')
    axis_KR_gau.set_xlabel('x')
    axis_KR_gau.set_ylabel('y')
    axis_KR_gau.set_xlim(20.63, 21.15)
    axis_KR_gau.set_ylim(39.00, 39.33)
    axis_KR_gau.contourf(gridx, gridy,  zstar,100, cmap=mycolor)
    KR_gau = axis_KR_gau.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_KR_gau.colorbar(KR_gau)
    st.write(fig_KR_gau)


#   This Part is the plotter of Kriging linear Interpolation 

    ok = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model = 'linear', #exponential, gaussian, linear
    verbose = True,
    enable_plotting = True
    )
    
    gridx = np.linspace(20.63, 21.15, 100)
    gridy = np.linspace(39.00, 39.33 ,100)
    zstar, ss = ok.execute('grid', gridx, gridy )


    fig_KR_lin,axis_KR_lin = plt.subplots()
    axis_KR_lin.set_title('Kriging Linear')
    axis_KR_lin.set_xlabel('x')
    axis_KR_lin.set_ylabel('y')
    axis_KR_lin.set_xlim(20.63, 21.15)
    axis_KR_lin.set_ylim(39.00, 39.33)
    axis_KR_lin.contourf(gridx, gridy,  zstar,100, cmap=mycolor)
    KR_lin = axis_KR_lin.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_KR_lin.colorbar(KR_lin)
    st.write(fig_KR_lin)


