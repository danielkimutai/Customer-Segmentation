import streamlit as st
import pandas as pd
import numpy
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
import joblib
# Load the pre-trained model
filename = 'dt_model.sav'
loaded_model=pickle.load(open(filename,'rb'))
df=pd.read_csv('Cluster_data.csv')



# Creating our app
nav=st.sidebar.radio("Navigation",["Home","Classifier"])

# Creating and Modifying our home page
if nav == "Home":
    st.title('Mall Customers Segmentation')
    st.header('Overview')
    st.write('A Company wants to increase their profits and  they need to find a way to give offers and promotions on their customers \n'
            'In order to assist I decided to build clustering model that segments into different segments in orde to targt \n'
            'them with different orders')
    st.subheader('Different Cluster Groups ')
    st.write('The clusters were grouped according to these groups:')
    st.markdown("Cluster 0: Middle Income, Middle Spending")
    st.markdown("Cluster 1: High Income,low Spending ")
    st.markdown("Cluster 2: Low Income, Low Spending ")
    st.markdown("Cluster 3: Low Income ,Low Spending ")
    st.markdown("Cluster 4: High Income ,High Spending ")
  
    
    # Plotting the different clusters visualizaton
    
    #st.set_option('deprecation.showPyplotGlobalUse', False)

    # Scale the data
   
    from sklearn.preprocessing import StandardScaler
    scaler= StandardScaler()
    scaled_features=scaler.fit_transform(df[['Annualincome','SpendingScore']])
    

    # return a label for each data point based on their cluster
    kmeans=KMeans(n_clusters=5,random_state=42)

    klabels=kmeans.fit_predict(scaled_features)
   
       
   # Creating a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plotting the clusters
    ax.scatter(scaled_features[klabels==0, 0], scaled_features[klabels==0, 1], s=50, color='green', label='cluster 0')
    ax.scatter(scaled_features[klabels==1, 0], scaled_features[klabels==1, 1], s=50, color='yellow', label='cluster 1')
    ax.scatter(scaled_features[klabels==2, 0], scaled_features[klabels==2, 1], s=50, color='blue', label='cluster 2')
    ax.scatter(scaled_features[klabels==3, 0], scaled_features[klabels==3, 1], s=50, color='red', label='cluster 3')
    ax.scatter(scaled_features[klabels==4, 0], scaled_features[klabels==4, 1], s=50, color='violet', label='cluster 4')

    # Visualizing the centroids
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')

    # Adding title and labels
    ax.set_title('Customer groups')
    ax.set_xlabel('Annual Income')
    ax.set_ylabel('Spending Score')
    ax.legend()

    # Passing the figure to Streamlit
    st.subheader('Distribution of clusters')
    st.pyplot(fig)

    # Count the number of instances in each cluster
    count = df['Cluster'].value_counts()

    x = count.index

    y = count.values


    # Create a figure and axis for the bar plot

    fig, ax = plt.subplots(figsize=(10, 5))


    # Plot the bar graph

    ax.bar(x, y)

    ax.set_title('Distribution of clusters', fontsize=16)

    ax.set_xlabel('Clusters')


    ax.set_ylabel('Distribution count')


    # Display the plot in the Streamlit app

    st.pyplot(fig)


    st.write("We can see that our data mainly consists of middle-income earners, followed closely by high spenders with high income. "

         "Customers who have low income and high spending habits are few compared to the rest.")
    
 ## Creating and Modifying our classifier   
else :
    st.text('Enter details here:')
    with st.form('my_form'):
        CustomerId=st.number_input(label='CustomerID',step=1)
        Gender=st.text_input(label='Gender(Male=1,Female=0)')
        Age=st.number_input(label='Age',step=1)
        AnnualIncome=st.number_input(label='Annualincome',step=1)
        SpendingScore=st.number_input(label='Spending Score',step=1)
        
        data=[[CustomerId,Gender,Age,AnnualIncome,SpendingScore]]
        
        Submitted=st.form_submit_button('Submit')
    if Submitted:
        clusters= loaded_model.predict(data)[0]
        if clusters == 0:
            st.success("Middle income ,Middle Spending Customer")
        elif clusters ==1 :
            st.success("High income,low Spending Customer")
        elif clusters== 2:
            st.success("Low income ,low Spending Customer")
        elif clusters ==3:
            st.success("Low income ,High spending Customer")
        else :
            st.success("High income ,High Spending Customer")
            
    