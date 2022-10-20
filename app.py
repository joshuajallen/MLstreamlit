import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


@st.cache(persist=True)
def load_data():
    data = pd.read_csv("mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

@st.cache(persist=True)
def split(df):
    y = df.type
    x = df.drop(columns = ['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    return x_train, x_test, y_train, y_test

def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if 'ROC curve' in metrics_list:
        st.subheader("ROC")
        plot_roc_curve(model, x_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if 'Precision curve' in metrics_list:
        st.subheader("Precision")
        plot_precision_recall_curve(model, x_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

def main():
    st.title("Credit scoring platform")
    st.sidebar.title("")
    st.markdown("This app assesses a users credit score based on selected inputs")

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']

    st.sidebar.subheader("Choose classifier:")
    classifier = st.sidebar.selectbox("Classifier", ("SVM", "Logistic", "RF"))

    if classifier == 'SVM':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.1, key='C')
        K = st.sidebar.radio("Kernal", ("rbf", "linear"), key="K")
        G = st.sidebar.radio("Gamma", ("scale", "auto"), key="G")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC curve", "Precision curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("SVM")
            model = SVC(C=C, kernel=K, gamma=G)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics_list=metrics, model=model, x_test=x_test, y_test=y_test, class_names=class_names)

    if classifier == 'Logistic':
        st.sidebar.subheader("Model Hyperparameters")
        C_LR = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.1, key='C_LR')
        max_iter = st.sidebar.slider("Max interations:", min_value=100, max_value=500, key="max_inter")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC curve", "Precision curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic")
            model = LogisticRegression(C=C_LR, max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics_list=metrics, model=model, x_test=x_test, y_test=y_test, class_names=class_names)

    if classifier == 'RF':
        st.sidebar.subheader("Model Hyperparameters")
        n_est = st.sidebar.number_input("Number of trees:", 100, 500, step=10, key="n_est")
        depth = st.sidebar.number_input("Max depth:", 1, 10, step=1, key="depth")
        bootstrap = st.sidebar.radio("Bootstrap:", ('True', 'False'), key="bootstrap")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC curve", "Precision curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("RF")
            model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics_list=metrics, model=model, x_test=x_test, y_test=y_test, class_names=class_names)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Data for classification:")
        st.write(df.head(5))


if __name__ == '__main__':
    main()
