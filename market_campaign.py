# ==========================================================# Import Libraries# ==========================================================

import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

def main():
    
    # ==========================================================# Load Dataset# ==========================================================

    #Loading the dataset
    data = pd.read_csv('Data/marketing_campaign.csv', sep="\t")
    print("Number of datapoints:", len(data))
    data.head()
    #Dropping the rows with missing values
    data = data.dropna()

    print("Number of datapoints after dropping missing values:", len(data))
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
    data.head()
    print("The newest customer joined on:", data['Dt_Customer'].max())
    print("The oldest customer joined on:", data['Dt_Customer'].min())
    #customer for each customer, we will calculate the number of days since they joined the company. 
    # #This will help us understand how long each customer has been with the company and may provide insights into their purchasing behavior.
    data['Customer_For'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days
    # print(data.loc[data['Customer_For'] == data['Customer_For'].min()])
    print("Total categories in the feature Martial_Status:", data['Marital_Status'].value_counts(), "\n")
    print("Total categories in the feature Education:", data['Education'].value_counts())
                                        #Grouping the data by Marital_Status and counting the number of customers in each group
    grouped_data = data.groupby('Marital_Status')['ID'].count()
    grouped_data.head()
    data.columns

    # ==========================================================# Feature Engineering# ==========================================================

    #Calculating the age of each customer based on their year of birth
    data['Age'] = 2026 - data['Year_Birth']

    #Calculating the total amount spent by each customer on different product categories
    data['total_spent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
    data.head()
    #Getting people who are living with a partner and those who are living alone based on their marital status.
    data['living_with_partner'] = data['Marital_Status'].replace(['Married', 'Together'], "Partner").replace(['Single', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], "Alone")
    #Getting the total number of children for each customer by summing the number of kids and teens in their household.
    data['Children'] = data['Kidhome'] + data['Teenhome']

    #Total number of family members in the household by adding the number of children and the number of adults (assuming that each customer is an adult)
    data['Family_Size'] = (data['living_with_partner'].map({'Partner': 2, 'Alone': 1}) + pd.to_numeric(data['Children']))

    #Feature pertaining parenthood
    data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

    print(data[['living_with_partner', 'Children', 'Family_Size']].drop_duplicates().head(20))
    data['Education'].value_counts()
    #The education feature has 6 categories, which can be grouped into 3 categories: Under Graduate, Graduate, and Post Graduate.
    data['Education'] = data['Education'].replace(['Basic', '2n Cycle'], "Under Graduate").replace(['PhD','Master'],"Post Graduate").replace(['Graduation'],"Graduate")

    # #Segmenting education levels in three groups
    # data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

    #For clarity and ease of analysis, we will rename the features that represent the amount spent on different product categories to more intuitive names.
    data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})


    # Droping the features that are not relevant for our analysis and modeling.
    data = data.drop(['Dt_Customer', 'Marital_Status', 'Z_CostContact', 'Z_Revenue', 'Year_Birth', 'ID'], axis=1)
    data.describe()

    # ==========================================================# Exploratory Data Analysis# ==========================================================

    #To plot some selected features

    #Setting up colors preferences
    sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
    pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
    cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

    #Plotting following features
    To_Plot = [ "Income", "Recency", "Customer_For", "Age", "total_spent", "Is_Parent"]
    print("Relative Plot Of Some Selected Features: A Data Subset")
    plt.figure()
    sns.pairplot(data[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))

    #Taking hue
    plt.show()
    #Dropping the outliers by setting a cap on Age and income.
    data = data[(data["Age"]<90)]
    data = data[(data["Income"]<600000)]
    print("The total number of data-points after removing the outliers are:", len(data))
    sns.pairplot(data[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))

    #Taking hue
    plt.show()
    data.info()
    #Get list of categorical variables
    s = (data.dtypes == 'str')
    object_cols = list(s[s].index)

    print("Categorical variables in the dataset:", object_cols)
    #Label Encoding the object dtypes.
    LE=LabelEncoder()
    for i in object_cols:
        data[i]=data[[i]].apply(LE.fit_transform)

    print("All features are now numerical")
    data.info()
    data.head()

    # ==========================================================# Data Preprocessing# ==========================================================

    #Creating a copy of the data to perform scaling and dimensionality reduction on it.
    ds = data.copy()

    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'Complain']
    ds = ds.drop(cols_del, axis=1)

    #scaling the data using StandardScaler to standardize the features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_data = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
    print("The data has been scaled and is ready for dimensionality reduction and clustering.")
    scaled_data.head()

    # ==========================================================# PCA# ==========================================================
    #Initializing PCA with 3 components to reduce the dimensionality of the data while retaining as much variance as possible.

    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    pca_components = pca.transform(scaled_data)
    pca_components

    pca_ds = pd.DataFrame(pca_components, columns=(["col1", "col2", "col3"]))
    pca_ds.describe().T
    #A 3D Projection Of Data In The Reduced Dimension
    x =pca_ds["col1"]
    y =pca_ds["col2"]
    z =pca_ds["col3"]
    #To plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x,y,z, c="maroon", marker="o" )
    ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
    plt.show()

    # ==========================================================# Clustering# ==========================================================

    # Quick examination of elbow method to find numbers of clusters to make. 
    # To identify the optimal number of clusters for KMeans clustering, we can use the Elbow Method.1 
    # The number of clusters at this point is considered optimal as it balances the trade-off between having too many clusters (overfitting) and too few clusters (underfitting).
    print('Elbow Method to determine the number of clusters to be formed:')
    Elbow_M = KElbowVisualizer(KMeans(), k=10)
    Elbow_M.fit(pca_ds)
    Elbow_M.show()
    #initializing the AgglomerativeClustering model with 4 clusters. 
    AC = AgglomerativeClustering(n_clusters=4)

    #fit Model and predict clusters

    yhat_AC = AC.fit_predict(pca_ds)

    pca_ds['Cluster'] = yhat_AC

    #Adding the cluster labels to the original dataset for further analysis and visualization.

    data['Cluster'] = yhat_AC
    data.head()
    #Plotting the clusters in 3D space to visualize how the data points are grouped based on the Agglomerative Clustering algorithm

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(111, projection='3d', label="bla")
    ax.scatter(x, y, z, s=40, c=pca_ds["Cluster"], marker='o', cmap = cmap )
    ax.set_title("The Plot Of The Clusters")
    plt.show()

    # ==========================================================# Visualizations# ==========================================================

    #sort by cluster
    data["Cluster"].value_counts().sort_index()
    pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
    pl = sns.countplot(x=data["Cluster"], palette= pal)
    pl.set_title("Distribution Of The Clusters")
    plt.show()
    data.columns
    pl = sns.scatterplot(data = data,x=data["total_spent"], y=data["Income"],hue=data["Cluster"], palette= pal)
    pl.set_title("Cluster's Profile Based On Income And Spending")
    plt.legend()
    plt.show()
    plt.figure()
    pl=sns.swarmplot(x=data["Cluster"], y=data["total_spent"], color= "#CBEDDD", alpha=0.5 )
    pl=sns.boxenplot(x=data["Cluster"], y=data["total_spent"], palette=pal)
    plt.show()
    plt.figure()
    pl=sns.swarmplot(x=data["Cluster"], y=data["Income"], color= "#CBEDDD", alpha=0.5 )
    pl=sns.boxenplot(x=data["Cluster"], y=data["Income"], palette=pal)
    plt.show()
    #Creating a feature to get a sum of accepted promotions
    data["Total_Promos"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]
    #Plotting count of total campaign accepted.
    plt.figure()
    pl = sns.countplot(x=data["Total_Promos"],hue=data["Cluster"], palette= pal)
    pl.set_title("Count Of Promotion Accepted")
    pl.set_xlabel("Number Of Total Accepted Promotions")
    plt.show()
    #Plotting the number of deals purchased
    plt.figure()
    pl=sns.boxenplot(y=data["NumDealsPurchases"],x=data["Cluster"], palette= pal)
    pl.set_title("Number of Deals Purchased")
    plt.show()
    data.columns
    Personal = [ "Kidhome","Teenhome","Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education","living_with_partner"]

    for i in Personal:
        plt.figure()
        sns.jointplot(x=data[i], y=data["total_spent"], hue= data["Cluster"], kind="kde", palette=pal)
        plt.show()


if __name__ == "__main__":
    main()