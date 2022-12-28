import streamlit as st
import pandas as pd 
import numpy as np

#from matplotlib import cm  #%matplotlib inline
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
mygreens = colors.LinearSegmentedColormap.from_list('mygreens', ['#000000', '#00FF00'])

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging


st.title('All interpolations')
st.text("This is a web app to explore interpolations from IRMA's Data")
uploaded_file = st.file_uploader('Upload your file here')

x_real = np.array([39.09933, 39.07888, 39.14904, 39.12208, 39.09518, 39.05061, 39.21634, 39.25915, 39.21899])
y_real = np.array([20.73085, 20.88525, 20.87591, 20.94730, 21.06071, 21.01207, 20.91295, 20.78422, 20.82493])

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

    x = data[:, 1]              # x coordinates 
    st.text(x)

    y = data[:, 2]              # y coordinates
    st.text(y)

    z = data[:, 5]
    st.text(z)
    
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title('The output of st.pyplot()')
    surf = ax1.scatter(y, x, 100, z, cmap=mygreens)
    fig1.colorbar(surf)
    st.pyplot(fig1)
    
    
    
    
    
    ok = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model = 'exponential', #exponential, gaussian, linear
    verbose = True,
    enable_plotting = True
    )
    
    gridx = np.linspace(39.05, 39.26, 100)
    gridy = np.linspace(20.73, 21.10 ,100)
    zstar, ss = ok.execute('grid', gridx, gridy )
 
    st.text(gridx)
    st.text(gridy)
    st.text(zstar)
    
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_title('The output of st.pyplot()')

    surf2 = ax2.contourf(gridy, gridx,  zstar,100, cmap=mygreens)
    fig2.colorbar(surf2)
    st.pyplot(fig2)
