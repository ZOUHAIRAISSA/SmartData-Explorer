import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score, mean_squared_error, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class AlgoML:
    def __init__(self):
        pass

    @staticmethod
    def preprocess_data(df, handle_missing_values=True, encode_categorical=True):
            # Initialize a dictionary to store original and encoded values
            encoding_dict = {}

            # Handle missing values
            if handle_missing_values:
                # Fill numerical columns with the mean
                for col in df.select_dtypes(include=[np.number]).columns:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mean())

                # Fill categorical columns with the mode
                for col in df.select_dtypes(include=[object]).columns:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mode()[0])

            # Encode categorical variables
            if encode_categorical:
                label_encoder = LabelEncoder()
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = label_encoder.fit_transform(df[col])
                    # Store the original and encoded values in the dictionary
                    encoding_dict[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

            return df, encoding_dict
    
    # @staticmethod
    # def extract_encoding_dict( df):
    #     if df is None:
    #         raise ValueError("DataFrame cannot be None")
        
    #     encoding_dict = {}
    #     label_encoder = LabelEncoder()

    #     for col in df.select_dtypes(include=['object']).columns:
    #         label_encoder.fit(df[col])
    #         encoding_dict[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    #         print(f"Encoded {col}: {encoding_dict[col]}")  # Debugging statement

    #     print("Encoding dictionary:", encoding_dict)  # Debugging statement
    #     return encoding_dict
    
    # @staticmethod
    # def preprocess_data(df, handle_missing_values=True, encode_categorical=True):
    #     # Handle missing values
    #     if handle_missing_values:
    #         # Fill numerical columns with the mean
    #         for col in df.select_dtypes(include=[np.number]).columns:
    #             if df[col].isnull().sum() > 0:
    #                 df[col] = df[col].fillna(df[col].mean())

    #         # Fill categorical columns with the mode
    #         for col in df.select_dtypes(include=[object]).columns:
    #             if df[col].isnull().sum() > 0:
    #                 df[col] = df[col].fillna(df[col].mode()[0])

    #     # Encode categorical variables
    #     if encode_categorical:
    #         label_encoder = LabelEncoder()
    #         for col in df.select_dtypes(include=['object']).columns:
    #             df[col] = label_encoder.fit_transform(df[col])

    #     return df

    # @staticmethod
    # def extract_encoding_dict(df):
    #     if df is None:
    #         raise ValueError("DataFrame cannot be None")
        
    #     encoding_dict = {}
    #     label_encoder = LabelEncoder()

    #     for col in df.select_dtypes(include=['object']).columns:
    #         label_encoder.fit(df[col])
    #         encoding_dict[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    #         print(f"Encoded {col}: {encoding_dict[col]}")  # Debugging statement

    #     print("Encoding dictionary:", encoding_dict)  # Debugging statement
    #     return encoding_dict

    
    # Function to apply machine learning algorithms based on the presence of a target variable
    def apply_ml_algorithms(self,df,target, test_size=0.3):

        results = {}
        # Check if the dataset has a target variable
        has_target = target in df.columns
        
        if has_target:
            X = df.drop(target, axis=1)
            y = df[target]
            
            # Determine if the target is categorical or continuous
            if y.nunique() <= 10:
                # Classification algorithms
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Decision Tree Classifier
                dtc = DecisionTreeClassifier()
                dtc.fit(X_train, y_train)
                dtc_score = dtc.score(X_test, y_test)
                results['Decision Tree Classifier'] = dtc_score
                
                # Random Forest Classifier
                rfc = RandomForestClassifier()
                rfc.fit(X_train, y_train)
                rfc_score = rfc.score(X_test, y_test)
                results['Random Forest Classifier'] = rfc_score
                
                # Logistic Regression
                lr = LogisticRegression()
                lr.fit(X_train, y_train)
                lr_score = lr.score(X_test, y_test)
                results['Logistic Regression'] = lr_score
                
                best_model = max(results, key=results.get)
                print('Best Classification Model:', best_model, results[best_model])
                
            else:
                # Regression algorithms
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Linear Regression
                lin_reg = LinearRegression()
                lin_reg.fit(X_train, y_train)
                lin_reg_score = lin_reg.score(X_test, y_test)
                results['Linear Regression'] = lin_reg_score
                
                # Decision Tree Regressor
                dtr = DecisionTreeRegressor()
                dtr.fit(X_train, y_train)
                dtr_score = dtr.score(X_test, y_test)
                results['Decision Tree Regressor'] = dtr_score
                
                # Random Forest Regressor
                rfr = RandomForestRegressor()
                rfr.fit(X_train, y_train)
                rfr_score = rfr.score(X_test, y_test)
                results['Random Forest Regressor'] = rfr_score
                
                best_model = max(results, key=results.get)
                print('Best Regression Model:', best_model, results[best_model])
                
        else:
            # Unsupervised learning algorithms
            # Clustering algorithms
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(df)
            kmeans_score = silhouette_score(df, kmeans.labels_)
            results['K-Means'] = kmeans_score
            
            dbscan = DBSCAN()
            dbscan.fit(df)
            dbscan_score = silhouette_score(df, dbscan.labels_) if len(set(dbscan.labels_)) > 1 else -1
            results['DBSCAN'] = dbscan_score
            
            # Dimensionality reduction algorithms
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df)
            pca_variance = pca.explained_variance_ratio_.sum()
            results['PCA Variance Explained'] = pca_variance
            

            best_model = max(results, key=results.get)
            print('Best Clustering Model:', best_model, results[best_model])
        
        return results
    
    # Regression
    @staticmethod
    def train_linear_regression(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_pred = lin_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print('Linear Regression MSE:', mse)
        return lin_reg
    
    @staticmethod
    def LinearRegressionModel(self,ds,tr,feat,inputs):
        # Specify target and features
        target = tr
        features = feat

        # Preprocess the data
        df = ds

        # Split the data into features (X) and target (y)
        X = df[features]
        y = df[target]

        # Train the linear regression model
        model = AlgoML.train_linear_regression(X, y)

        # Example form input for prediction
        form_input =  [float(i) for i in inputs]   

        # Predict the target value using the trained model
        prediction = AlgoML.predict_with_model(model, form_input)
        return prediction
    
    @staticmethod
    def RandomForestRegressionModel(self, ds, tr, feat, inputs):
        # Specify target and features
        target = tr
        features = feat

        # Preprocess the data
        df = ds

        # Split the data into features (X) and target (y)
        X = df[features]
        y = df[target]

        # Train the random forest regressor
        model = RandomForestRegressor()
        model.fit(X, y)

        # Example form input for prediction
        form_input = [float(i) for i in inputs]

        # Predict the target value using the trained model
        prediction = AlgoML.predict_with_model(model, form_input)
        return prediction

    @staticmethod
    def DecisionTreeRegressionModel(self, ds, tr, feat, inputs):
        # Specify target and features
        target = tr
        features = feat

        # Preprocess the data
        df = ds

        # Split the data into features (X) and target (y)
        X = df[features]
        y = df[target]

        # Train the decision tree regressor
        model = DecisionTreeRegressor()
        model.fit(X, y)

        # Example form input for prediction
        form_input = [float(i) for i in inputs]

        # Predict the target value using the trained model
        prediction = AlgoML.predict_with_model(model, form_input)
        return prediction
    
    # Classification

    @staticmethod
    def DecisionTreeClassifierModel(self, ds, tr, feat, inputs):
        # Specify target and features
        target = tr
        features = feat

        # Preprocess the data
        df = ds

        # Split the data into features (X) and target (y)
        X = df[features]
        y = df[target]

        # Train the decision tree classifier
        model = DecisionTreeClassifier()
        model.fit(X, y)

        # Example form input for prediction
        form_input = [float(i) for i in inputs]

        # Predict the target value using the trained model
        prediction = AlgoML.predict_with_model(model, form_input)
        return prediction
    
    @staticmethod
    def RandomForestClassifierModel(self, ds, tr, feat, inputs):
        # Specify target and features
        target = tr
        features = feat

        # Preprocess the data
        df = ds

        # Split the data into features (X) and target (y)
        X = df[features]
        y = df[target]

        # Train the random forest classifier
        model = RandomForestClassifier()
        model.fit(X, y)

        # Example form input for prediction
        form_input = [float(i) for i in inputs]

        # Predict the target value using the trained model
        prediction = AlgoML.predict_with_model(model, form_input)
        return prediction

    @staticmethod
    def LogisticRegressionModel(self, ds, tr, feat, inputs):
        # Specify target and features
        target = tr
        features = feat

        # Preprocess the data
        df = ds

        # Split the data into features (X) and target (y)
        X = df[features]
        y = df[target]

        # Train the logistic regression model
        model = LogisticRegression()
        model.fit(X, y)

        # Example form input for prediction
        form_input = [float(i) for i in inputs]

        # Predict the target value using the trained model
        prediction = AlgoML.predict_with_model(model, form_input)
        return prediction

    @staticmethod
    def predict_with_model(model, features):
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return prediction
    

            
            