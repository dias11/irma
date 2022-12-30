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

from scipy.interpolate import RBFInterpolator

from decimal import Decimal

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
    
    
#---------------------- KRIGING INTERPOLATOR---------------------   
    
    
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





#---------------------- RBF INTERPOLATOR---------------------



#   This Part is the plotter of Rbf - Linear -r

    cords = np.vstack((x, y)).T
    gridx = np.linspace(20.63, 21.15 ,100)
    gridy = np.linspace(39.00, 39.33, 100)
    xv, yv = np.meshgrid(gridx, gridy)
    mesh = np.array(np.meshgrid(gridx,gridy)).T.reshape(-1,2)  #makes all coordinatios combinations
    yflat = RBFInterpolator(cords, z, kernel='linear')
    ok = yflat(mesh)
    
    fig_Rbf_lin,axis_Rbf_lin = plt.subplots()
    axis_Rbf_lin.set_title('Rbf Linear -r')
    axis_Rbf_lin.set_xlabel('x')
    axis_Rbf_lin.set_ylabel('y')
    axis_Rbf_lin.set_xlim(20.63, 21.15)
    axis_Rbf_lin.set_ylim(39.00, 39.33)
    axis_Rbf_lin.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_lin = axis_Rbf_lin.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_lin.colorbar(Rbf_lin)
    st.write(fig_Rbf_lin)


#   This Part is the plotter of Rbf - thin_plate_spline r**2 * log(r)    
    
    yflat = RBFInterpolator(cords, z, kernel='thin_plate_spline')    
    ok = yflat(mesh)
    
    fig_Rbf_spline,axis_Rbf_spline = plt.subplots()
    axis_Rbf_spline.set_title('Rbf thin_plate_spline r**2 * log(r)')
    axis_Rbf_spline.set_xlabel('x')
    axis_Rbf_spline.set_ylabel('y')
    axis_Rbf_spline.set_xlim(20.63, 21.15)
    axis_Rbf_spline.set_ylim(39.00, 39.33)
    axis_Rbf_spline.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_spline = axis_Rbf_spline.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_spline.colorbar(Rbf_spline)
    st.write(fig_Rbf_spline)    
    
    
    
#   This Part is the plotter of Rbf - cubic  r**3   
    
    yflat = RBFInterpolator(cords, z, kernel='cubic')    
    ok = yflat(mesh)    
    
    fig_Rbf_cubic,axis_Rbf_cubic = plt.subplots()
    axis_Rbf_cubic.set_title('Rbf cubic r**3')
    axis_Rbf_cubic.set_xlabel('x')
    axis_Rbf_cubic.set_ylabel('y')
    axis_Rbf_cubic.set_xlim(20.63, 21.15)
    axis_Rbf_cubic.set_ylim(39.00, 39.33)
    axis_Rbf_cubic.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_cubic = axis_Rbf_cubic.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_cubic.colorbar(Rbf_cubic)
    st.write(fig_Rbf_cubic)        
    
    
    
    
#   This Part is the plotter of Rbf - quintic  r**5       
    
    yflat = RBFInterpolator(cords, z, kernel='quintic')    
    ok = yflat(mesh)       
    
    fig_Rbf_quintic,axis_Rbf_quintic = plt.subplots()
    axis_Rbf_quintic.set_title('Rbf quintic -r**5')
    axis_Rbf_quintic.set_xlabel('x')
    axis_Rbf_quintic.set_ylabel('y')
    axis_Rbf_quintic.set_xlim(20.63, 21.15)
    axis_Rbf_quintic.set_ylim(39.00, 39.33)
    axis_Rbf_quintic.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_quintic = axis_Rbf_quintic.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_quintic.colorbar(Rbf_quintic)
    st.write(fig_Rbf_quintic)         
    
    
#   This Part is the plotter of Rbf - multiquadric    

    yflat = RBFInterpolator(cords, z, kernel='multiquadric', epsilon=9.5)    
    ok = yflat(mesh)    
    
    fig_Rbf_multiq,axis_Rbf_multiq = plt.subplots()
    axis_Rbf_multiq.set_title('Rbf multiquadric -sqrt(1 + r**2) epsilon=9.5')
    axis_Rbf_multiq.set_xlabel('x')
    axis_Rbf_multiq.set_ylabel('y')
    axis_Rbf_multiq.set_xlim(20.63, 21.15)
    axis_Rbf_multiq.set_ylim(39.00, 39.33)
    axis_Rbf_multiq.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_multiq = axis_Rbf_multiq.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_multiq.colorbar(Rbf_multiq)
    st.write(fig_Rbf_multiq)      
    
    
#   This Part is the plotter of Rbf - inverse_multiquadric     
    
    yflat = RBFInterpolator(cords, z, kernel='inverse_multiquadric', epsilon=9.5)    
    ok = yflat(mesh)   

    fig_Rbf_invMult,axis_Rbf_invMult = plt.subplots()
    axis_Rbf_invMult.set_title('Rbf inverse_multiquadric 1/sqrt(1 + r**2) epsilon=9.5')
    axis_Rbf_invMult.set_xlabel('x')
    axis_Rbf_invMult.set_ylabel('y')
    axis_Rbf_invMult.set_xlim(20.63, 21.15)
    axis_Rbf_invMult.set_ylim(39.00, 39.33)
    axis_Rbf_invMult.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_invMult = axis_Rbf_invMult.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_invMult.colorbar(Rbf_invMult)
    st.write(fig_Rbf_invMult)      


#   This Part is the plotter of Rbf - inverse_quadratic     

    yflat = RBFInterpolator(cords, z, kernel='inverse_quadratic', epsilon=5.5)    
    ok = yflat(mesh)       
    
    fig_Rbf_invQuadr,axis_Rbf_invQuadr = plt.subplots()
    axis_Rbf_invQuadr.set_title('Rbf inverse_quadratic : 1/(1 + r**2) epsilon=5.5')
    axis_Rbf_invQuadr.set_xlabel('x')
    axis_Rbf_invQuadr.set_ylabel('y')
    axis_Rbf_invQuadr.set_xlim(20.63, 21.15)
    axis_Rbf_invQuadr.set_ylim(39.00, 39.33)
    axis_Rbf_invQuadr.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_invQuadr = axis_Rbf_invQuadr.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_invQuadr.colorbar(Rbf_invQuadr)
    st.write(fig_Rbf_invQuadr)       
    
    
 #   This Part is the plotter of Rbf - Gaussian 
    
    yflat = RBFInterpolator(cords, z, kernel='gaussian', epsilon=9.5)    
    ok = yflat(mesh)       
    
    fig_Rbf_Gaus,axis_Rbf_Gaus = plt.subplots()
    axis_Rbf_Gaus.set_title('Rbf gaussian : exp(-r**2) epsilon=9.5')
    axis_Rbf_Gaus.set_xlabel('x')
    axis_Rbf_Gaus.set_ylabel('y')
    axis_Rbf_Gaus.set_xlim(20.63, 21.15)
    axis_Rbf_Gaus.set_ylim(39.00, 39.33)
    axis_Rbf_Gaus.scatter(xv, yv, 100, ok, cmap=mycolor)
    Rbf_Gaus = axis_Rbf_Gaus.scatter(x, y, 100, z, cmap=mycolor, edgecolors='black')
    fig_Rbf_Gaus.colorbar(Rbf_Gaus)
    st.write(fig_Rbf_Gaus)      
    
    
    
    
    
