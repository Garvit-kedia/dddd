import streamlit as st 
import pandas as pd 
import plotly.express as px 
import os 
import warnings
import datetime
import plotly.graph_objects as go 
import numpy as np
import seaborn as sns
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scipy.stats as stats
from PIL import Image 
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder

def create_heatmap(df):
     
     df_encoded = df.copy()
     label_encoder = LabelEncoder()
     for column in df_encoded.columns:
      df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
     corr_matrix=df_encoded.corr()
     heatmap=ff.create_annotated_heatmap(
          z=corr_matrix.values,
          x=list(corr_matrix.columns),
          y=list(corr_matrix.index),
          annotation_text=corr_matrix.round(2).values,
          colorscale="Viridis",
          showscale=True
     )
     heatmap.update_layout(
          title="correaltion heatmap",
          xaxis_title="Features",
          template="plotly_white"
     )
     return heatmap
     


def create_boxplot(df,boxcolumn):
     fig=px.box(
          df,
          y=boxcolumn,
          title=f"Outlier Detection:{boxcolumn}",
          points="outliers",
          color_discrete_sequence=["#FF5733"]

     )
     fig.update_layout(
          yaxis_title=boxcolumn,
          template="plotly_white",
          hovermode="x"
     )
     return fig

def create_scatter(df,xaxis,yaxis):
     fig=px.scatter(
          df,
          x=xaxis,
          y=yaxis,
          color=xaxis,
          color_continuous_scale='viridis',
        #   title=f"scatter plot betwwn {xaxis} and {yaxis}",
          labels={xaxis:xaxis,yaxis:yaxis},
          hover_data={xaxis:True,yaxis:True}
          
     )
     fig.update_layout(
          xaxis_title=xaxis,
          yaxis_title=yaxis,
          template="plotly_white",
          hovermode="closest"
     )
     return fig
     
def create_histogram(df, column):
    
   # Plotly Histogram with Hover Effects
    fig = px.histogram(
        df, 
        x=a, 
        nbins=30, 
        title=f"Distribution of {a}",
        histnorm="density",  # Normalize histogram
        opacity=0.7,
        color_discrete_sequence=["#636EFA"],  # Custom color
        hover_data={a: True}  # Show values on hover
    )

  

  
    # Update Layout
    fig.update_layout(
        xaxis_title=a,
        yaxis_title="Density",
        hovermode="x",
        bargap=0.05,
        template="plotly_white"
    )

    return fig

               
                                                                 

def create_pie_plot(df, column_to_plot):
     pie_fig=px.pie(df,names=column_to_plot,hole=0.3)
     return pie_fig


st.set_page_config(page_title="Data Analysis",page_icon=":bar_chart",layout="wide")
st.title("Automted Data Analysis")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
image=Image.open('logo_path.jpg')
col1,col2=st.columns([0.1,0.9])
with col1:
     st.image(image,width=100)
html_title = """
    <style>
    .title-test {
    font-weight:bold;
    padding:5px;
    border-radius:6px;
    }
    </style>
    <center><h1 class="title-test">Data analysis Dashboard</h1></center>"""
with col2:
    st.markdown(html_title, unsafe_allow_html=True)
data=st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))

if data is not None:
        if data.name.endswith('xlsx'):
            df = pd.read_excel(data)
        else:
            df = pd.read_csv(data)
            for column in df.columns:
                df[column] = df[column].fillna(df[column].mode()[0])  # Fill NaN with the most frequent value

        st.dataframe(df.head())
        numerical_cols=df.columns.to_list()
        hist_c=df.select_dtypes(include=['int', 'float']).columns.tolist()

        col3,col4,col5=st.columns([0.45,0.45,0.45])
        with col3:
             xaxis,yaxis=st.multiselect("",numerical_cols,key="Xaxisselt",default=numerical_cols[:2])
             fig=create_scatter(df,xaxis,yaxis)
             st.plotly_chart(fig)
             
        with col4:
             xaxis,yaxis=st.multiselect("",numerical_cols,key="Xaxilt",default=numerical_cols[:2])
      
             fig=px.bar(df,x=xaxis,y=yaxis,labels={"TotalSales" :"total Sales {$}"},
                        title=f"{xaxis} by {yaxis}",hover_data=[yaxis],
                        template="gridon",height=500)
             st.plotly_chart(fig,use_container_width=True)
        # _, view1,dwn1=st.columns([0.15,0.20,0.20])
        # with view1:
        #      expander=st.expander(xaxis)
        #      data=df[[xaxis,yaxis]].groupby(by=xaxis)[yaxis].sum()
        #      expander.write(data)
        # with dwn1:
        #      st.download_button("Get Data",data=data.to_csv().encode("utf-8"),
        #                         file_name=f"{xaxis}.csv",mime="text/csv")
        with col5:
             
             a=st.selectbox("selct numerical columns",hist_c)
             
             hist_fig=create_histogram(df,a)
             st.plotly_chart(hist_fig)

        col6,col7,col8=st.columns([0.45,0.45,0.45])
        with col6:
             
             unique_value_counts = df.nunique()
             filtered_columns = unique_value_counts[unique_value_counts < 12].index.tolist()
             all_columns = df.columns.to_list()
             column_to_plot = st.selectbox("" ,filtered_columns)
             pie_fig = create_pie_plot(df, column_to_plot)
             st.plotly_chart(pie_fig)
        with col7:
             boxcolumn = st.selectbox("Select a numerical column for outlier detection", numerical_cols)
             fig=create_boxplot(df,boxcolumn)
             st.plotly_chart(fig)
        fig=create_heatmap(df)
        st.plotly_chart(fig)
        
            

             


             

