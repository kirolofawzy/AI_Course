<h1 style="font-size:40px; font-family:Segoe UI; color:#1565C0; font-weight:bold;">
❤️ Heart Failure Disease Prediction Using Machine Learning & Genetic Algorithm
</h1>

<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
📌 Project Overview
</h2>

<p style="font-size:18px; font-family:Segoe UI; color:#333;">
This project aims to predict heart failure disease using seven Machine Learning classification algorithms. 
After that, a Genetic Algorithm (GA) is applied to reduce the number of features, improve efficiency, and compare performance before and after GA.
</p>


<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
📊 Dataset Features
</h2>

<h3 style="font-size:20px; font-family:Segoe UI; color:#1E88E5;">
Dataset contains 13 clinical features:
</h3>

<pre style="background:#f4f4f4; padding:10px; font-size:16px; font-family:Consolas;">
['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
 'restecg', 'thalach', 'exang', 'oldpeak',
 'slope', 'ca', 'thal']
</pre>


<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
🤖 Machine Learning Models Used
</h2>

<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">1️⃣ Logistic Regression</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">2️⃣ K-Nearest Neighbors (KNN)</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">3️⃣ Support Vector Machine (SVM)</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">4️⃣ Decision Tree Classifier</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">5️⃣ Random Forest Classifier</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">6️⃣ Naive Bayes</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">7️⃣ Artificial Neural Network (ANN)</h3>


<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
🧬 Genetic Algorithm Feature Reduction
</h2>

<p style="font-size:18px; font-family:Segoe UI; color:#333;">
Genetic Algorithm (GA) was used to reduce the number of features for each ML model, 
improving the efficiency while keeping or improving performance.
</p>

<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">
Feature Reduction Results:
</h3>

<pre style="background:#f4f4f4; padding:10px; font-size:16px; font-family:Consolas;">
LogisticRegression: 6 features
SVM: 6 features
KNN: 7 features
RandomForest: 5 features
DecisionTree: 4 features
NaiveBayes: 8 features
ANN: 8 features
</pre>

<p style="font-size:18px; font-family:Segoe UI; color:#333;">
Each algorithm produces:
<ul>
<li>GA_&lt;Model&gt;_train.csv</li>
<li>GA_&lt;Model&gt;_test.csv</li>
</ul>
</p>


<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
📈 Model Performance Before GA
</h2>

<pre style="background:#f4f4f4; padding:10px; font-size:16px; font-family:Consolas;">
Model       Accuracy Precision Recall F1-Score
log_reg       79.51     80.23   79.51   79.38
SVM           80.49     81.68   80.49   80.29
KNN           95.61     95.97   95.61   95.60
DecisionTree  98.54     98.58   98.54   98.54
RandomForest  98.54     98.58   98.54   98.54
NaiveBayes    80.00     81.05   80.00   79.82
ANN           93.66     93.66   93.66   93.66
</pre>


<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
📈 Model Performance After GA
</h2>

<pre style="background:#f4f4f4; padding:10px; font-size:16px; font-family:Consolas;">
Model           Accuracy Precision Recall F1-Score
log_reg_GA        82.44     82.63   82.44   82.41
SVM_GA            82.44     82.74   82.44   82.39
KNN_GA            99.02     99.04   99.02   99.02
DT_Model_GA      100.00    100.00  100.00  100.00
RF_Model_GA      100.00    100.00  100.00  100.00
nb_Model_GA       82.93     83.08   82.93   82.90
NN_Model_GA       89.27     89.51   89.27   89.25
</pre>


<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
📉 Data Visualization
</h2>

<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">✔ Correlation heatmap</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">✔ Feature distribution</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">✔ Outlier detection</h3>
<h3 style="font-size:22px; font-family:Segoe UI; color:#1E88E5;">✔ Class imbalance analysis</h3>

<h2 style="font-size:30px; font-family:Segoe UI; font-weight:bold; color:#0D47A1;">
✔ Conclusion
</h2>

<p style="font-size:18px; font-family:Segoe UI; color:#333;">
The combination of Machine Learning and Genetic Algorithm significantly improved model accuracy 
and reduced the complexity of heart disease prediction. This makes the system more efficient and suitable for real medical applications.
</p>

<h2 style="font-size:32px; font-family:Segoe UI; font-weight:bold; color:#0D47A1; margin-bottom:20px;">
👨‍💻 Project Team
</h2>

<div style="
    display: flex; 
    flex-wrap: wrap; 
    gap: 20px; 
    padding: 10px;
">

    <!-- Member 1 -->
    <div style="
        background: #ffffff; 
        border-radius: 12px; 
        padding: 20px; 
        width: 280px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-left: 6px solid #1565C0;
    ">
        <h3 style="font-size:22px; font-family:Segoe UI; color:#1565C0; margin-top:0;">
            ⭐ Kirolos Fawzy Kamel
        </h3>
        <p style="font-size:16px; font-family:Segoe UI; color:#333;">
            Machine Learning Engineer  
            <br>GA Optimization & Model Evaluation
        </p>
    </div>

    <!-- Member 2 -->
    <div style="
        background: #ffffff; 
        border-radius: 12px; 
        padding: 20px; 
        width: 280px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-left: 6px solid #00897B;
    ">
        <h3 style="font-size:22px; font-family:Segoe UI; color:#00897B; margin-top:0;">
            ⭐ 3abdallah Salah Elsayd
        </h3>
        <p style="font-size:16px; font-family:Segoe UI; color:#333;">
            Data Preprocessing &  
            <br>Medical Feature Analysis
        </p>
    </div>

    <!-- Member 3 -->
    <div style="
        background: #ffffff; 
        border-radius: 12px; 
        padding: 20px; 
        width: 280px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-left: 6px solid #6A1B9A;
    ">
        <h3 style="font-size:22px; font-family:Segoe UI; color:#6A1B9A; margin-top:0;">
            ⭐ Maxim Mamdouh Salib
        </h3>
        <p style="font-size:16px; font-family:Segoe UI; color:#333;">
            Visualization & Data Insights  
            <br>Correlation & Risk Pattern Analysis
        </p>
    </div>

    <!-- Member 4 -->
    <div style="
        background: #ffffff; 
        border-radius: 12px; 
        padding: 20px; 
        width: 280px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-left: 6px solid #C62828;
    ">
        <h3 style="font-size:22px; font-family:Segoe UI; color:#C62828; margin-top:0;">
            ⭐ Petter Nader Halem
        </h3>
        <p style="font-size:16px; font-family:Segoe UI; color:#333;">
            Model Training Pipeline  
            <br>Accuracy Benchmarking
        </p>
    </div>

</div>




