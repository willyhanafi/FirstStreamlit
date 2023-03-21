import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.experimental import enable_iterative_imputer
from autoviz.classify_method import data_cleaning_suggestions ,data_suggestions



def main():
    st.title("")
    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

    # Create a sidebar with a dropdown menu of CSV files
    selected_csv = st.sidebar.selectbox('Select a CSV file', csv_files)

    # Read the selected CSV file and display it in the main Streamlit area
    if selected_csv:
        df = pd.read_csv(selected_csv)
        st.write(f"Data {selected_csv}")
        st.dataframe(df.head(200))
    
    if selected_csv == 'melbourne_houses.csv':
        X = df.drop('Price',axis=1)
        y = df['Price']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    st.write ('jumlah baris dan kolom : ', X.shape)
    st.write('Jumlah Kelas : ', len(np.unique(y)))
    
    #unique_y = y.unique()
    #unique_y.sort()
    #st.write(f"Nama Kelas : {unique_y}")
    # Replace the values in y with their corresponding class labels
    #y = y.replace(dict(enumerate(unique_y)))
    
    def plot_metrics(asd):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Data Null' in eda:
            datanull = df.isnull().sum().sort_values(ascending=False)
            st.write("Data Null : ")
            st.write(datanull)
        if 'Correlation Matrix' in eda:
            matrixcorr = df.corr()[y.name]
            #top15 adalah attribut yang menampung 15 attribut dengan korelasi tertinggi terhadap harga 
            top15 = matrixcorr.nlargest(16)[1:].index
            #Merge the 15 top attribut with label
            top_df = df[top15].join(df[y.name])
            corr = top_df.corr()
            plt.figure(figsize=(15,12))
            sns.heatmap(corr, cmap="coolwarm", annot=True)
            st.pyplot()


    eda = st.sidebar.multiselect("Exploration Data Analysis",
                                     ('Data Null','Correlation Matrix'))
    plot_metrics(eda)
    

    option = st.sidebar.selectbox("Pilih Pre-Processing Data",("None","Data Cleaning + Transformation", "Data Transformation"))
    if option == "None":
        if selected_csv == 'melbourne_houses.csv':
            X = df.drop('Price',axis=1)
            y = df['Price']
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
    elif option == "Data Cleaning + Transformation":
        from sklearn.preprocessing import LabelEncoder
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            # Replace missing values with "missing"
            df[col] = df[col].fillna("missing")
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer()
        imputed_data = imputer.fit_transform(df)
        df = pd.DataFrame(imputed_data, columns=df.columns)
        datanull = df.isnull().sum().sort_values(ascending=False)
        st.write("Jumlah Data Yang Null Setelah Transformasi : ")
        st.write(datanull.head(20))
        st.write("Jenis Data Setelah Transformasi : ")
        st.write(df.dtypes)
    else : 
        from sklearn.preprocessing import LabelEncoder
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        datanull = df.isnull().sum().sort_values(ascending=False)
        st.write("Jumlah Data Yang Null Setelah Transformasi : ")
        st.write(datanull.head(20))
        st.write("Jenis Data Setelah Transformasi : ")
        st.write(df.dtypes)

    num_classes = y.nunique()
    if num_classes > 10:
        task = 'regression'
    else:
        task = 'classification'
   
    if task == "classification":
        from pycaret.classification import setup, predict_model, create_model, plot_model, evaluate_model
    elif task == "regression":
        from pycaret.regression import setup, predict_model, create_model, plot_model, evaluate_model

    setupmodel = st.sidebar.checkbox('Setup Model')

    if setupmodel:
        set = setup(X,target=y ,session_id=42,train_size= 0.8)
        st.write(set)
        st.success("Setup Model Success")
        model = st.sidebar.selectbox("Pilih Model",("None","Extreme Gradient Boosting",
                                                    "Light Gradient Boosting Machine", 
                                                    "Random Forest Regressor"))
        if model == "None":
            st.write(" ")
        elif model == "Extreme Gradient Boosting":
            algo = create_model('xgboost')
            st.success("Model Extreme Gradient Berhasil Dibuat")
        elif model == "Light Gradient Boosting Machine":
            algo = create_model('lightgbm')
            st.success("Model Light Gradient Boosting Berhasil Dibuat")
        else : 
            algo = create_model('rf')
            st.success("Model Random Forest Berhasil Dibuat")


        plot = st.sidebar.selectbox("Plot Model",
                                        ('None','Manifold Learning','Prediction Error Plot','Recursive Feat. Selection'))

        if plot == 'Manifold Learning' :
            plot_model(algo, plot = 'manifold',display_format='streamlit')
        if plot =='Prediction Error Plot' :
            plot_model(algo, plot = 'error',display_format='streamlit')
        if plot =='Recursive Feat. Selection':
            plot_model(algo, plot = 'rfe',display_format='streamlit') 
        else : 
            None   

        pred = st.sidebar.checkbox('Prediksi Data')

        if pred:
            preds = predict_model(algo)
            st.dataframe(preds)
            st.warning("untuk klasifikasi, hasil prediksinya yaitu antara rentang 0 sampai 1. artinya, apabila skornya 1 itu berarti keyakinannya 100% terhadap kelas asli, apabila 0.71 berarti 71% keyakinannya terhadap kelas asli. \n hasil prediksi ada di kolom paling kanan ")
    # Create a dropdown for model selection
    
if __name__ == '__main__':
    main()
