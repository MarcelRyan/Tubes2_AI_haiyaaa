import numpy as np

class NaiveBayesClassifier:
    
    def __init__(self):
        pass
    
    # Separate the data into subset data for each column
    def separate_feature(self, X, y):
        """
        X : array, list of all columns except target and its value
        y : array, list of target column value
        
        return : Classify which rows belongs to certain y value
        """
        
        # Make dictionary with y value as keys, and X as values
        separated_feature = {}
        for i in range(len(X)):
            column_values = X[i]
            feature_name = y[i]
        
            if feature_name not in separated_feature:
                separated_feature[feature_name] = []
            separated_feature[feature_name].append(column_values)
        
        return separated_feature

    # Function to search for mean and std for every column in X
    def statistics_info(self, X):

        """
        Calculates mean and standard deviation for each column
        X : the data to be calculated

        return : mean and standard deviation of each column
        """
        
        for column in zip(*X):  # Unpack the data first and then search for every column std and mean
            yield {
                'std' : np.std(column),
                'mean' : np.mean(column),
            }
    
    def fit(self, X, y):
        """
        Training the model
        X : array, list of all columns except target and its value
        y : array, list of target column value
        
        Returns : Dictionary containing prior probability (probability of getting a certain target value without any evidence), mean, and std of every column
        """
        
        separated_features = self.separate_feature(X, y)
        self.column_info = {}
        
        for feature_name, column_values in separated_features.items():
            self.column_info[feature_name] = {
                "prior_probability" : len(column_values)/len(X),
                "statistics" : [i for i in self.statistics_info(column_values)],
            }
        
        return self.column_info

    def gaussian_distribution(self, x, mean, std):
        
        """
        Calculation of predicted values ​​using the Gaussian distribution formula on one of the attributes
        x : variable value
        mean : the average attribute value of the variable
        std : standard deviation of the variable attribute

        return : predicted value
        """
        
        exponent = np.exp(-((x - mean)**2 / (2 * std**2)))
        
        return exponent/(std * np.sqrt(2*np.pi))

    def predict(self, X):
        
        """
        Prediction calculations using all attributes
        X : data used to make predictions

        return : prediction result data for each row
        """

        hypotheses = []
        
        for row in X:
            
            joint_probability = {}
            
            for feature_name, columns in self.column_info.items():
                total_columns = len(columns['statistics'])
                likelihood = 1
                
                for idx in range(total_columns):
                    feature_value = row[idx]
                    mean = columns['statistics'][idx]['mean']
                    std = columns['statistics'][idx]['std']
                    normal_probability = self.gaussian_distribution(feature_value, mean, std)
                    likelihood *= normal_probability
                
                prior_probability = columns['prior_probability']
                joint_probability[feature_name] = prior_probability * likelihood
            
            hypothesis = max(joint_probability, key=joint_probability.get)
            hypotheses.append(hypothesis)
        
        return hypotheses
    

    # Calculate model accuracy
    def accuracy(self, y_test, y_pred):

        """
        Calculation of suitability of predicted values ​​and validation values
        y_test : validation data target values
        y_pred : training data target values

        return : accuracy between training predictions and validation values
        """

        true = 0
        
        for y_t, y_p in zip(y_test, y_pred):
            if y_t == y_p:
                true += 1
        
        return true / len(y_test)
    
        
    
        