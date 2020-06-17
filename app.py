import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from PIL import Image




def main():
    html_head = """
        	<div style="background-color:#5752ff;">
        	<p style="color:white;font-size:50px;padding:20px;padding-bottom:0px;font-weight:bold">
        	 THE BINARY CLASSIFIER 
        	</p>
        	<p style="color:white;font-size:20px;margin-left:20px;padding-bottom:10px" >
        	&nbsp&nbsp&nbsp&nbsp👉  An Auto ML WEB-APP  🤖</p>
        	</div>"""

    st.markdown(html_head, unsafe_allow_html=True)

    st.subheader("An Web-App for Machine-Learning Enthusiasts !!")

    st.subheader("Made with ❤️ For Data Science Community !!! ")

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    # ------sidebar-header parrt----------

    html_head = """
        	<div style="background-color:#5752ff;">
        	<p style="color:white;font-size:20px;padding:7px; text-align:Center;font-weight:bold;">
        	THE BINARY CLASSIFIER <br> 
        	Web App
        	</p></div>
        	"""

    st.sidebar.markdown(html_head, unsafe_allow_html=True)
    st.sidebar.markdown("")
    st.sidebar.markdown("")

    # -----side-bar harder over-----------

    # ----introduction-------------


    st.subheader(" About THE BINARY CLASSIFIER AUTO ML WEB-APP :")

    whatis = """
                	
                	<div style="background-color:#ff7817;">

                	
                	<p style="color:white;font-size:15px;padding:17px; text-align:justify;font-weight:bold;">
                	 The Binary Classifier ML app allows you to Train and Test classification machine learning models
                	 through your own dataset. <br><br>
                	 This Web-app also includes tutorials for the ML Classifier models used by you, so it becomes easy for
                	 an learning developer to get the knowledge of BINARY CLASSIFICATION through ML.
                	<br><br>
                	Moreover, This app can also be useful in order to select the appropriate model for your dataset,
                	by trying and testing ML models through various hyper-parameters, 
                	which is the main motive after builiding this Web-App.<br>
                	You can get more information about the models from the tutorial section below.
                	<br><br> 
                	The BINARY CLASSIFIACTION MODELS USED IN THIS APP ARE :
                	<br>👉  Support Vector Classifier
                	<br>👉  Logistic Regression Classifier
                	<br>👉  Random Forest Classifier        
                	</p>
                	
                	</div>
                	
                	"""

    st.markdown(whatis, unsafe_allow_html=True)


#----------------tutorials------------------

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader(" Tutorials of Binary Classifiers used in this Web-App 👇")


    tutorial = st.selectbox("select a Classifier for tutorial : ",
                                              ["Select a Classifier",
                                               "Support Vector Machine (SVM)",
                                               "Logistic Regression Classifier",
                                               "Random Forest Trees Classifier"])
    if tutorial == "Support Vector Machine (SVM)":
        SVM = """
                            	<div style="background-color:#eb3b9e;">
                            	<p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                            	Generally, Support Vector Machines is considered to be a classification approach, 
                            	it but can be employed in both types of classification and regression problems. 
                            	It can easily handle multiple continuous and categorical variables. 
                            	<br>SVM constructs a hyperplane in multidimensional space to separate different classes. 
                            	SVM generates optimal hyperplane in an iterative manner, which minimizes the errors. 
                            	<br>The core idea of SVM is to find a maximum marginal hyperplane(MMH) 
                            	that best divides the dataset into classes.
                            	</p></div>
                            	"""
        st.markdown("")

        st.markdown("")
        st.header("Support Vector Machine (SVM) Binary Classifier")
        st.markdown(SVM, unsafe_allow_html=True)
        svm1 = Image.open("images/svm1.png").convert("RGB")
        st.image(svm1, caption="SVM Refrence", use_column_width=False)

        SVM1 = """
                <div style="background-color:#eb3b9e;">
                <p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                The main objective is to segregate the given dataset in the best possible way.<br> 
                The distance between the either nearest points is known as the margin.<br> 
                The objective is to select a hyperplane with the maximum possible margin 
                between support vectors in the given dataset. <br><br>
                SVM searches for the maximum marginal hyperplane in the following steps:<br><br>
                1) Generate hyperplanes which segregates the classes in the best way. 
                Left-hand side figure showing three hyperplanes black, blue and orange. 
                Here, the blue and orange have higher classification error, 
                but the black is separating the two classes correctly.
                <br><br>
                2) Select the right hyperplane with the maximum segregation from 
                the either nearest data points as shown in the right-hand side figure.
                </p></div>
                
                """
        st.markdown("")
        st.subheader("How SVM Classifier works:")
        st.markdown(SVM1, unsafe_allow_html=True)
        svm1 = Image.open("images/svm2.png").convert("RGB")
        st.image(svm1, use_column_width=False)

        SVM2 = """
                <div style="background-color:#eb3b9e;">
                <p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                <b>Kernel:</b> The main function of the kernel is to transform the given dataset input data into the required form. 
                There are various types of functions such as linear, polynomial, and radial basis function (RBF). 
                Polynomial and RBF are useful for non-linear hyperplane. 
                Polynomial and RBF kernels compute the separation line in the higher dimension. 
                In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. 
                This transformation can lead to more accurate classifiers.
                <br><br>
                <b>Regularization ( C ):</b> Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. 
                Here C is the penalty parameter, which represents misclassification or error term. 
                The misclassification or error term tells the SVM optimization how much error is bearable. 
                This is how you can control the trade-off between decision boundary and misclassification term. 
                A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
                <br><br>
                <b>Gamma :</b> A lower value of Gamma will loosely fit the training dataset, 
                whereas a higher value of gamma will exactly fit the training dataset, 
                which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line,
                while the a value of gamma considers all the data points in the calculation of the separation line.
                </p></div>

                """
        st.markdown("")
        st.subheader("SVM Classifier important Tuning Parameters to be used:")
        st.markdown(SVM2, unsafe_allow_html=True)



    #-----------logistic regression-------------


    if tutorial == "Logistic Regression Classifier":
        logistic = """
                            	<div style="background-color:#eb3b9e;">
                            	<p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                            	Logistic regression is a statistical method for predicting binary classes. 
                            	The outcome or target variable is dichotomous in nature. Dichotomous means there are only two possible classes. 
                            	For example, it can be used for cancer detection problems. It computes the probability of an event occurrence.
                            	<br><br>
                                It is a special case of linear regression where the target variable is categorical in nature. 
                                It uses a log of odds as the dependent variable. 
                                Logistic Regression predicts the probability of occurrence of a binary event utilizing a logit function.
                                <br><br>
                                <b>Types of Logistic Regression :</b>
                                <br>
                                <b>Binary Logistic Regression:</b> The target variable has only two possible outcomes such as Spam or Not Spam, Cancer or No Cancer.
                                <br><b>Multinomial Logistic Regression:</b> The target variable has three or more nominal categories such as predicting the type of Wine.
                                <br><b>Ordinal Logistic Regression:</b> the target variable has three or more ordinal categories such as restaurant or product rating from 1 to 5.
                                </p></div>
                            	"""
        st.markdown("")

        st.markdown("")
        st.header("Logistic Regression Binary Classifier")
        st.markdown(logistic, unsafe_allow_html=True)
        logistic1 = """
                        <div style="background-color:#eb3b9e;">
                        <p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                        Linear regression gives you a continuous output, but logistic regression provides a constant output. 
                        An example of the continuous output is house price and stock price. 
                        Example's of the discrete output is predicting whether a patient has cancer or not, predicting whether the customer will churn. 
                        Linear regression is estimated using Ordinary Least Squares (OLS) while 
                        logistic regression is estimated using Maximum Likelihood Estimation (MLE) approach.
                        </p></div>
                        """

        st.markdown("")
        st.subheader("Linear Regression v/s Logistic Regression")
        st.markdown(logistic1, unsafe_allow_html=True)
        logistic = Image.open("images/logistic1.png").convert("RGB")
        st.image(logistic, use_column_width=False)

        logistic2 = """
                        <div style="background-color:#eb3b9e;">
                        <p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                        <b>Max_iter :</b> Maximum number of iterations taken for the solvers to converge.
                        <br><br>
                        <b>Regularization ( C ) :</b> Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. 
                        Here C is the penalty parameter, which represents misclassification or error term. 
                        The misclassification or error term tells the SVM optimization how much error is bearable. 
                        This is how you can control the trade-off between decision boundary and misclassification term. 
                        A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
                        <br><br>
                        </p></div>

                        """
        st.markdown("")
        st.subheader("Logistic regression Classifier important Tuning Parameters to be used :")
        st.markdown(logistic2, unsafe_allow_html=True)





    #----------Random forest------------------

    if tutorial == "Random Forest Trees Classifier":
        random = """
                            	<div style="background-color:#eb3b9e;">
                            	<p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                            	 Random forests is a supervised learning algorithm. It can be used both for classification and regression. 
                            	 It is also the most flexible and easy to use algorithm. A forest is comprised of trees. 
                            	 It is said that the more trees it has, the more robust a forest is. 
                            	 <br>Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting. 
                            	 It also provides a pretty good indicator of the feature importance.
                            	 <br><br>
                            	 It technically is an ensemble method (based on the divide-and-conquer approach) of decision trees generated on a randomly split dataset. 
                            	 This collection of decision tree classifiers is also known as the forest.<br>
                            	 The individual decision trees are generated using an attribute selection indicator such as information gain, gain ratio, and Gini index 
                            	 for each attribute. Each tree depends on an independent random sample. 
                            	 In a classification problem, each tree votes and the most popular class is chosen as the final result. 
                            	 <br><br>In the case of regression, the average of all the tree outputs is considered as the final result. 
                            	 It is simpler and more powerful compared to the other non-linear classification algorithms
                            	</p></div>
                            	"""
        st.markdown("")

        st.markdown("")
        st.header("Random Forest Trees Binary Classifier")
        st.markdown(random, unsafe_allow_html=True)


        random1 = """
                        <div style="background-color:#eb3b9e;">
                        <p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                        <b>It works in four steps :</b>
                        <br>
                        1. Select random samples from a given dataset.
                        <br>2. Construct a decision tree for each sample and get a prediction result from each decision tree.
                        <br>3. Perform a vote for each predicted result.
                        <br>4. Select the prediction result with the most votes as the final prediction.
                        </p></div>
                        """

        st.markdown("")
        st.subheader("How Random Forest Classifier Works :")
        st.markdown(random1, unsafe_allow_html=True)
        logistic = Image.open("images/random1.jpg").convert("RGB")
        st.image(logistic, use_column_width=False)

        random2 = """
                        <div style="background-color:#eb3b9e;">
                        <p style="color:white;font-size:17px;padding:20px; text-align:justify;">
                        <b>n_estimators :</b> The number of trees in the forest.
                        <br><br>
                        <b>max_depth :</b> The maximum depth of the tree. If None, then nodes are expanded until 
                        all leaves are pure or until all leaves contain less than min_samples_split samples.
                        <br><br>
                        <b>Bootstrap :</b> Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree..

                        </p></div>

                        """
        st.markdown("")
        st.subheader("Random Forest Classifier important Tuning Parameters to be used :")
        st.markdown(random2, unsafe_allow_html=True)


    #--------- how to use-----------------

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader(" How to use this Binary Classifier Web-app 👇")

    selectservices = """              	
                            <p style="background-color:#cf005d;color:white;font-size:15px;padding:20px; text-align:justify;font-weight:bold;">
                            1. &nbsp Select or upload dataset ( CSV Format) of your choice.
                            <br>2. &nbsp Choose a column ( column should contain only 2 unique values ) from your dataset.
                            <br>3. &nbsp Then, choose the corresponding classifier from side-menu.
                            <br>4. &nbsp Select Hyperparameters of the Classifiaction model. 
                            <br>5. &nbsp Train the model , by selecting the checkbox. 
                            <br>6. &nbsp Observe the results of ACCURACY,  PRECISION & RECALL. 
                            <br>7. &nbsp Select the metrics you want to get from the trained and tested Classifier model.
                            <br>8. &nbsp click the checkbox after to see the testing data of model and its coressponding prediction.  

                            </p> 
                    	    """
    st.markdown(selectservices, unsafe_allow_html=True)

    # ------introduction over---------



    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.header("Binary Classifiaction Model Training / Testing with Dataset : (PRACTICAL APPROACH)")
    st.subheader("UPLOAD THE DATASET .")

    data = st.file_uploader("Upload a Dataset in [ CSV ] format only.", type=["csv"])

#-------data fetching from csv-----

    if data is not None:
        original_data = pd.read_csv(data)

        st.markdown("")

        if st.checkbox("Select to See your uploaded / Original Dataset", False):
            st.subheader("This is the original form of your Dataset : ")
            st.dataframe(original_data)
            st.markdown("")
            st.markdown("")
        st.markdown("")

        st.markdown("")

        st.subheader("Select the column from following , for Classifiaction / Prediction :")

        st.warning(
            "NOTE :  \n This is a binary Classification Web-app, Thus it only predicts accurately between Two things "
            "for example: YES/NO or TRUE/FALSE or other two unique values.  \n Select such a column from the dataset which has only two unique values to Predict / Classify "
            "OR the model won't train.")

        original_column_names = original_data.columns.tolist()
        original_column = st.selectbox("Select the column from your dataset , that you want to classify OR Predict :",
                                       original_column_names)

        original_class_names = [original_data[original_column].unique().tolist()]

        if len(original_class_names[0]) <= 2:
            st.write("Your Outcome can be from the following items :", original_data[original_column].unique())
            # st.warning(" If, '0' = NO / FALSE and '1' = YES / TRUE for the output.")
        else:
            st.write("Your Selected column has more than 2 unique values :", original_data[original_column].unique())

            st.warning("Please select such a column from your dataset inorder to perform binary classification.")
        st.markdown("")
        st.markdown("")

        data = original_data

        if st.checkbox("Select to see Summary of your dataset"):
            st.subheader("This is the summary of your Dataset : ")
            st.write(original_data.describe())
            st.markdown("")
            st.markdown("")

        st.markdown("")


#-----------data transformation for input to model----------

    def load_data(data):

        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])

        if st.checkbox("See Raw ( Classification ) Data", False):
            st.subheader("This is the Raw ( Classification ) form of Dataset : ")
            st.warning("NOTE : All the words will be sorted alphabetically in "
                       "Raw view of dataset. This dataset is used to feed the "
                       "data to the model in order to train and test the model.  ")
            st.dataframe(data)
            st.markdown("")

        return data

    if data is not None:
        classify_data = load_data(data)



    # -------creating test and train dataset----------
    def split_dataset(classify_data):

        if classify_data is not None:

            X = classify_data.drop(original_column, axis=1)
            y = classify_data[original_column]

            x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
            # # st.write(x_train )
            # # x_train = x_train.values.reshape(-1,1)
            # # y_train = y_train.to_numpy().reshape(-1,1)
            # st.write(x_train,type(x_train),x_train.shape)
            # st.write(y_train,type(y_train),y_train.shape)
            # st.write(type(x_test))
            # st.write(type(y_test))
            # # np.array(x_train)
            # # st.write("x train",type(x_train),x_train.shape)

        return x_train, x_test, y_train, y_test, original_column


    if data is not None:
        x_train, x_test, y_train, y_test, classify_column = split_dataset(classify_data)
        # st.write(x_train, x_test, y_train, y_test)


    # ----metrics plotting-----

    def plot_metrics(metrics_list):

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix Plotting : ")

            plot_confusion_matrix(model, x_test, y_test, display_labels=original_class_names[0])
            st.pyplot()

            st.markdown("")
            st.markdown("")

        if 'ROC Curve' in metrics_list:
            if len(original_class_names[0]) <= 2:
                st.subheader("ROC Curve Plotting : ")
                plot_roc_curve(model, x_test, y_test)
                st.pyplot()

                st.markdown("")
                st.markdown("")

            else:
                st.subheader("ROC Curve Plotting : Can not be plotted ")
                st.warning(
                    "This is binary classification application and ROC Curve Plotting can be done when there are 2 values to predict / classify ")

        if 'Precision-Recall Curve' in metrics_list:
            if len(original_class_names[0]) <= 2:
                st.subheader("Precision-Recall Curve Plotting : ")

                plot_precision_recall_curve(model, x_test, y_test)
                st.pyplot()

                st.markdown("")
                st.markdown("")

            else:
                st.subheader("ROC Curve Plotting : Can not be plotted ")
                st.warning(
                    "This is binary classification application and precision - recall Curve Plotting can be done when there are 2 values to predict / classify ")


                # ----metrics plotting over-----

        st.markdown("")

    # ---------classifier selection side bar--------


    selectclassifier = """
                        <div style="background-color:purple;">
                    	<p style="color:white;font-size:15px;padding:5px; text-align:center;">
                    	Select the classifier from below after selecting the dataset.
                    	</p></div>
                    	"""
    st.sidebar.markdown(selectclassifier, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    selectclassifier = """
                        <div style="background-color:purple;">
                        <p style="background-color:purple;color:white;font-size:15px;padding:15px; text-align:justify;">
                    	👈 Select the classifier from the side menu after selecting / uploading the dataset.
                    	</p></div>
                    	"""
    st.markdown(selectclassifier, unsafe_allow_html=True)

    # --------classifier selection over-----


    # --------All Classifiers ----------------


    if data is not None:
        class_names = [classify_data[original_column].unique().tolist()]

    selectedclassifier = st.sidebar.selectbox("select a Classifier : ",
                                              ["Select a Classifier",
                                               "Support Vector Machine (SVM)",
                                               "Logistic Regression Classifier",
                                               "Random Forest Trees Classifier"])

    if data is not None:

        # -------------SVM--------------------

        if selectedclassifier == "Support Vector Machine (SVM)":


            if len(original_class_names[0]) <= 2:
                st.sidebar.subheader("Select below Model Hyperparameters for "
                                     "Support Vector Machine [ SVM ]")
                C = st.sidebar.number_input("C ( Regularization Paramter ) ", 0.1, 10.0, step=0.5, key="C")
                kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
                gamma = st.sidebar.radio("Gamma ( Kernel Co-efficient )", ("scale", "auto"), key="gamma")

                if st.sidebar.checkbox("Train the model", key="Train"):
                    st.markdown("")
                    st.markdown("")
                    st.subheader("MODEL TRAINING PHASE : ")
                    st.warning(
                        "SVM Binary Classification model training has been started with your uploaded dataset .....")

                    model = SVC(C=C, kernel=kernel, gamma=gamma)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_predict = model.predict(x_test)

                    if accuracy < 0.30:
                        st.warning(
                            "The SVM Binary Classification model has been trained but you have selected the label from the dataset which"
                            " has more than Two values / labels to predict.")
                        st.error(
                            "Please upload proper dataset OR Select the Column from the dataset which has only Two values / labels to predict "
                            "in order to perform Binary Classification through SVM.")

                    else:
                        st.success(
                            " The SVM Binary Classification Model has been trained SUCCESSFULY !! CONGRATULATIONS")
                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TRAINING / TESTING PHASE RESULTS : ")

                        st.write("Accuracy : ", accuracy.round(3))
                        st.write("Precision : ", precision_score(y_test, y_predict, average='micro').round(3))
                        st.write("Recall : ",
                                 recall_score(y_test, y_predict, pos_label='positive', average='micro').round(3))

                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TRAINING / TESTING PHASE Metrics plots : ")

                        metrics = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
                        metrics_list = st.multiselect(
                            "Select which type of metrics you want to plot for model Training Phase.", metrics)
                        if st.button("Create Metrics plots :"):
                            plot_metrics(metrics_list)

                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TESTING PHASE DATA : ")
                        if st.checkbox("Click here to see classification the model based on testing data :"):
                            label = original_column + "[model output]"
                            test_data = x_test
                            test_data[label] = y_predict
                            # test_data = np.where(test_data==0,"NO.",test_data)
                            st.write("Output of the model for test inputs. "
                                     "(The output of your selected column would be the last column in the below testing dataset.)",
                                     test_data)
                            st.warning(" WHERE '0' = NO / FALSE and  '1' = YES / TRUE")

            else:
                st.sidebar.warning(
                    "Choose such a column from dataset which has only Two unique values to classify & to train the model.")








                # ------------------------Logistic Regression----------------


        elif selectedclassifier == "Logistic Regression Classifier":

            Logistic = """
                    	<div style="background-color:#00b3ff;">
                    	<p style="color:white;font-size:23px;padding:7px; text-align:Center;font-weight:bold;">
                    	Hello, Welcome to<br>
                    	THE BINARY CLASSIFIER  
                    	Web App
                    	</p></div>
                    	"""

            st.markdown(Logistic, unsafe_allow_html=True)

            if len(original_class_names[0]) <= 2:

                st.sidebar.subheader("Select below Model Hyperparameters for "
                                     "Logistic Regression")
                C = st.sidebar.number_input("C ( Regularization Paramter ) ", 0.1, 10.0, step=0.5, key="C")
                max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")

                if st.sidebar.checkbox("Train the Logistic Regression Classifier model", key="Train"):
                    st.markdown("")
                    st.markdown("")
                    st.subheader("MODEL TRAINING PHASE : ")
                    st.warning(
                        "Logistic Regression Binary Classification model training has been started with your uploaded dataset .....")

                    model = LogisticRegression(C=C, max_iter=max_iter)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_predict = model.predict(x_test)

                    if accuracy < 0.40:
                        st.warning(
                            "The Logistic Regression Binary Classification model has been trained but you have selected the label from the dataset which"
                            " has more than Two values / labels to predict.")
                        st.error(
                            "Please upload proper dataset OR Select the Column from the dataset which has only Two values / labels to predict "
                            "in order to perform Binary Classification through Logistic Regression.")

                    else:
                        st.success(
                            " The Logistic Regression Binary Classification Model has been trained SUCCESSFULY !! CONGRATULATIONS")
                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TRAINING PHASE RESULTS : ")
                        st.write("Accuracy : ", accuracy.round(3))
                        st.write("Precision : ", precision_score(y_test, y_predict, average='micro').round(3))
                        st.write("Recall : ",
                                 recall_score(y_test, y_predict, pos_label='positive', average='micro').round(3))

                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TRAINING PHASE Metrics plots : ")

                        metrics = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
                        metrics_list = st.multiselect(
                            "Select which type of metrics you want to plot for model Training Phase.", metrics)
                        if st.button("Create Metrics plots :"):
                            plot_metrics(metrics_list)

                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TESTING PHASE DATA : ")

                        if st.checkbox(
                                "Click here to see the test data feed to the model and its corresponding output of the model :"):
                            label = original_column + "[model output]"
                            test_data = x_test
                            test_data[label] = y_predict
                            # test_data = np.where(test_data==0,"NO.",test_data)
                            st.write("Output of the model for test inputs. "
                                     "(The output of your selected column would be the last column in the below testing dataset.)",
                                     test_data)
                            st.warning(" WHERE '0' = NO / FALSE and '1' = YES / TRUE for the output.")








            else:
                st.sidebar.warning(
                    "Choose such a column from dataset which has only Two unique values to classify & to train the model.")





        # ---------------------Random forest trees classifier--------------
        elif selectedclassifier == "Random Forest Trees Classifier":

            if len(original_class_names[0]) <= 2:

                st.sidebar.subheader("Select below Model Hyperparameters for "
                                     "Random Forest Trees Classifier")

                n_estimators = st.sidebar.number_input("The number of the trees in the forest", 100, 5000, step=10,
                                                       key="n_estimators")
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key="max_depth")
                bootstrap = st.sidebar.radio("Bootstrap samples When buillding the Random Forest Classifier ",
                                             ("True", "Flase"), key="bootstrap")

                if st.sidebar.checkbox("Train the Random Forest Trees Classifier model", key="Train"):
                    st.markdown("")
                    st.markdown("")
                    st.subheader("MODEL TRAINING PHASE : ")
                    st.warning(
                        "Random Forest Trees Classifier model training phase has been started with your uploaded dataset .....")

                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                                   n_jobs=-1)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_predict = model.predict(x_test)

                    if accuracy < 0.40:
                        st.warning(
                            "The Random Forest Trees Classifier model has been trained but you have selected the label from the dataset which"
                            " has more than Two values / labels to predict.")
                        st.error(
                            "Please upload proper dataset OR Select the Column from the dataset which has only Two values / labels to predict "
                            "in order to perform Binary Classification through Logistic Regression.")

                    else:
                        st.success(
                            " The Random Forest Trees Classifier Model has been trained SUCCESSFULY !! CONGRATULATIONS")
                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TRAINING PHASE RESULTS : ")
                        st.write("Accuracy : ", accuracy.round(3))
                        st.write("Precision : ", precision_score(y_test, y_predict, average='micro').round(3))
                        st.write("Recall : ",
                                 recall_score(y_test, y_predict, pos_label='positive', average='micro').round(3))

                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TRAINING PHASE Metrics plots : ")

                        metrics = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
                        metrics_list = st.multiselect(
                            "Select which type of metrics you want to plot for model Training Phase.", metrics)
                        if st.button("Create Metrics plots :"):
                            plot_metrics(metrics_list)

                        st.markdown("")
                        st.markdown("")

                        st.subheader("MODEL TESTING PHASE DATA : ")

                        if st.checkbox(
                                "Click here to see the test data feed to the model and its corresponding output of the model :"):
                            label = original_column + "[model output]"
                            test_data = x_test
                            test_data[label] = y_predict
                            # test_data = np.where(test_data==0,"NO.",test_data)
                            st.write("Output of the model for test inputs. "
                                     "(The output of your selected column would be the last column in the below testing dataset.)",
                                     test_data)
                            st.warning(" WHERE '0' = NO / FALSE and '1' = YES / TRUE for the output.")




            else:
                st.sidebar.warning(
                    "Choose such a column from dataset which has only Two unique values to classify & to train the model.")



    else:
        st.sidebar.warning("PLEASE UPLOAD THE DATASET TO RUN THE MODEL TRAINING / TESTING PHASE:")



    #-------------- credits-----------
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("")

    st.markdown("")
    st.markdown("")
    st.markdown("")

    customize = """
                                	<div style="background-color:#ff3b3b;">
                                	<p style="color:white;font-size:25px;padding:10px;text-align:Center;">
                                	Made by SOHAM SHAH</p>
                                	<p style="color:white;font-size:20px;padding:10px;padding-top:0px;text-align:Center;font-weight:bold;">
                                    Visit my <a style="color:blue;font-size:20px;text-align:Center;" href="https://github.com/soham2512">GitHub</a> !!!
                                	<br>
                                	Lets Connect on <a style="color:blue;font-size:20px;text-align:Center;" href="https://www.linkedin.com/in/soham-shah-669636139/">Linked-in</a> !!!
                                	</p>
                                	</div>
                                	"""

    st.markdown(customize, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
