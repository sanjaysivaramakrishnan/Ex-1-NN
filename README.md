<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\Neural_networks\Datasets\Churn_Modelling.csv')
df.head()
### Finding Missing Values
df.isnull().sum()
#Handling Missing values
# As this value doesn't contains any null values we don't want to worry about handling missing values
# Check for duplicates
duplicates = df.duplicated()
duplicates
print(f'Number of duplicates rows: {duplicates.sum()}')    
#Detect Outliers
df.describe()
df.head()
df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
#Normalize the dataset
Scaler = MinMaxScaler()
df1 = pd.DataFrame(Scaler.fit_transform(df))
df1
X = df1.iloc[:,:-1].values
X
y = df1.iloc[:,-1].values
y
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)
X_train
X_test

```


## OUTPUT:
![Screenshot 2025-03-10 063914](https://github.com/user-attachments/assets/157a24b4-6693-495d-97a4-b765b5538570)
![Screenshot 2025-03-10 063931](https://github.com/user-attachments/assets/1cb5a8f7-6f9a-4804-b5f9-0207ca68a531)
![Screenshot 2025-03-10 063948](https://github.com/user-attachments/assets/d9207824-d6a5-4f4b-b622-c2b738eb06f2)
![Screenshot 2025-03-10 064046](https://github.com/user-attachments/assets/ab5227cd-aa6f-46be-982c-a125a27bbaef)
![Screenshot 2025-03-10 064113](https://github.com/user-attachments/assets/9bfb8e94-167e-43ce-8922-5f784ec2015d)

![Screenshot 2025-03-10 064130](https://github.com/user-attachments/assets/2a83aad2-3460-44a4-91ac-599f274d810c)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


