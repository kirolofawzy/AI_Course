from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from deap import base, creator, tools, algorithms
from collections import Counter
import random
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


import tensorflow as tf

tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print("TensorFlow/Keras loaded successfully")


print("Loading dataset...")
df = pd.read_csv("../heart.csv")
x = df.drop('target', axis=1)
y = df['target']
feature_names = x.columns.tolist()

print(f"\n Dataset Information:")
print(f"   Total samples: {len(df)}")
print(f"   Features: {len(feature_names)}")
print(f"   Feature names: {feature_names}")


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values
n_features = X_train_np.shape[1]

print(f"\n   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")
print(f"   Number of features: {n_features}")



class ANNClassifier(BaseEstimator, ClassifierMixin):


    def __init__(self, epochs=100, batch_size=32, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.input_dim = None

    def create_model(self, input_dim):

        model = Sequential([
            Input(shape=(input_dim,)),
            BatchNormalization(),

            Dense(256, activation='relu'),
            Dropout(0.4),
            BatchNormalization(),

            Dense(128, activation='relu'),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        """تدريب النموذج"""
        self.input_dim = X.shape[1]
        self.model = self.create_model(self.input_dim)


        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        self.training_history_ = history.history
        return self

    def predict(self, X):
        """التنبؤ"""
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """احتمالات التنبؤ"""
        predictions = self.model.predict(X, verbose=0)
        return np.hstack([1 - predictions, predictions])

    def evaluate(self, X, y):
        """تقييم النموذج"""
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        return loss, accuracy



models_config = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=SEED, max_iter=1000),
        'penalty_factor': 0.01,
        'ga_params': {'n_gen': 8, 'pop_size': 15}
    },
    'SVM': {
        'model': SVC(kernel='linear', random_state=SEED),
        'penalty_factor': 0.01,
        'ga_params': {'n_gen': 8, 'pop_size': 15}
    },
    'KNN': {
        'model': KNeighborsClassifier(n_neighbors=2),
        'penalty_factor': 0.02,
        'ga_params': {'n_gen': 10, 'pop_size': 15}
    },
    'RandomForest': {
        'model': RandomForestClassifier(n_estimators=7, random_state=SEED),
        'penalty_factor': 0.005,
        'ga_params': {'n_gen': 6, 'pop_size': 15}
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=SEED),
        'penalty_factor': 0.015,
        'ga_params': {'n_gen': 10, 'pop_size': 15}
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'penalty_factor': 0.005,
        'ga_params': {'n_gen': 8, 'pop_size': 15}
    },
    'ANN': {
        'model': ANNClassifier(epochs=100, batch_size=32, verbose=0),
        'penalty_factor': 0.01,
        'ga_params': {'n_gen': 6, 'pop_size': 10, 'cxpb': 0.7, 'mutpb': 0.3}
    }
}

print(f"\nModels to be optimized: {list(models_config.keys())}")


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def clone_model(model):
    """استنساخ النموذج"""
    import copy
    return copy.deepcopy(model)


def create_ga_toolbox(model_name, model_obj, penalty_factor=0.01):
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if model_name == 'ANN':
        def fitness_function(ind):
            mask = np.array(ind) == 1
            selected_features = mask.sum()

            if selected_features == 0:
                return (0.001,)

            try:
                X_train_selected = X_train_np[:, mask]
                X_test_selected = X_test_np[:, mask]

                model_copy = ANNClassifier(epochs=50, batch_size=32, verbose=0)
                model_copy.fit(X_train_selected, y_train_np)
                pred = model_copy.predict(X_test_selected)
                accuracy = accuracy_score(y_test_np, pred)

                penalty = penalty_factor * (selected_features / n_features) * 0.5
                adjusted_accuracy = accuracy * (1 - penalty)

                return (max(0.001, min(0.999, adjusted_accuracy)),)
            except Exception:
                return (0.001,)
    else:
        def fitness_function(ind):
            mask = np.array(ind) == 1
            selected_features = mask.sum()

            if selected_features == 0:
                return (0.001,)

            try:
                X_train_selected = X_train_np[:, mask]
                X_test_selected = X_test_np[:, mask]

                model_copy = clone_model(model_obj)
                model_copy.fit(X_train_selected, y_train_np)
                pred = model_copy.predict(X_test_selected)
                accuracy = accuracy_score(y_test_np, pred)

                penalty = penalty_factor * (selected_features / n_features)
                adjusted_accuracy = accuracy * (1 - penalty)

                return (max(0.001, min(0.999, adjusted_accuracy)),)
            except Exception:
                return (0.001,)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_ga_for_model(model_name, model_config):
    print(f"\n{'=' * 60}")
    print(f" Running GA for {model_name}")
    print('=' * 60)

    toolbox = create_ga_toolbox(
        model_name,
        model_config['model'],
        model_config['penalty_factor']
    )

    ga_params = model_config.get('ga_params', {'n_gen': 8, 'pop_size': 15})
    n_gen = ga_params['n_gen']
    pop_size = ga_params['pop_size']
    cxpb = ga_params.get('cxpb', 0.8)
    mutpb = ga_params.get('mutpb', 0.2)

    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(3)

    try:
        population, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=n_gen,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        best_individual = hof[0]
        best_mask = np.array(best_individual) == 1
        best_features = X_train.columns[best_mask].tolist()

        X_train_selected = X_train_np[:, best_mask]
        X_test_selected = X_test_np[:, best_mask]

        model = clone_model(model_config['model'])

        if model_name == 'ANN':
            model.epochs = 200
            model.fit(X_train_selected, y_train_np)
            pred = model.predict(X_test_selected)
        else:
            model.fit(X_train_selected, y_train_np)
            pred = model.predict(X_test_selected)

        true_accuracy = accuracy_score(y_test_np, pred)

        print(f"\n {model_name} Results:")
        print(f"   Selected features: {len(best_features)}")
        print(f"   Accuracy: {true_accuracy:.4f}")
        print(f"   Features: {best_features}")

        return {
            'best_individual': best_individual,
            'best_mask': best_mask,
            'best_features': best_features,
            'num_features': len(best_features),
            'accuracy': true_accuracy,
            'hof': hof,
            'logbook': logbook
        }

    except Exception as e:
        print(f"Error in GA for {model_name}: {e}")
        return None



print(f"\n{'=' * 70}")
print("GENETIC ALGORITHM FEATURE SELECTION")
print('=' * 70)

results = {}
failed_models = []

for model_name, config in models_config.items():
    try:
        result = run_ga_for_model(model_name, config)
        if result is not None:
            results[model_name] = result
        else:
            failed_models.append(model_name)
    except Exception as e:
        print(f"Failed to run GA for {model_name}: {e}")
        failed_models.append(model_name)

print(f"\nCompleted GA for {len(results)} out of {len(models_config)} models")
if failed_models:
    print(f"   Failed models: {failed_models}")

if results:
    print(f"\n{'=' * 70}")
    print("FINAL FEATURE SELECTION RESULTS")
    print('=' * 70)

    results_list = []
    for model_name, result in results.items():
        results_list.append({
            'Model': model_name,
            'Num_Features': result['num_features'],
            'Accuracy': result['accuracy'],
            'Features': ', '.join(result['best_features'][:5]) +
                        ('...' if len(result['best_features']) > 5 else '')
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    print(f"\nModel Comparison (Sorted by Accuracy):")
    print(results_df.to_string(index=False))

    all_features_list = []
    for result in results.values():
        all_features_list.extend(result['best_features'])

    feature_counts = Counter(all_features_list)

    print(f"\nFeature Selection Frequency:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        models_with_feature = [name for name, res in results.items()
                               if feature in res['best_features']]
        print(f"   {feature}: {count} models ({', '.join(models_with_feature[:3])}" +
              ("..." if len(models_with_feature) > 3 else "") + ")")

    common_threshold = max(1, len(results) // 2)
    high_freq_features = [feat for feat, count in feature_counts.items()
                          if count >= common_threshold]

    if high_freq_features:
        print(f"\nHighly Recommended Features (selected by ≥{common_threshold} models):")
        for feat in high_freq_features:
            print(f"   ✓ {feat}")

    if 'ANN' in results:
        print(f"\n{'=' * 70}")
        print("TRAINING FINAL ANN WITH SELECTED FEATURES")
        print('=' * 70)

        ann_features = results['ANN']['best_features']
        ann_accuracy = results['ANN']['accuracy']

        if ann_features:
            X_train_ann = X_train[ann_features].values
            X_test_ann = X_test[ann_features].values

            print(f"Training ANN with {len(ann_features)} features: {ann_features}")
            print(f"Expected accuracy from GA: {ann_accuracy:.4f}")

            NN_Model_GA = Sequential([
                Input(shape=(X_train_ann.shape[1],)),
                BatchNormalization(),

                Dense(256, activation='relu'),
                Dropout(0.4),
                BatchNormalization(),

                Dense(128, activation='relu'),
                Dropout(0.3),

                Dense(64, activation='relu'),
                Dropout(0.2),

                Dense(1, activation='sigmoid')
            ])

            NN_Model_GA.compile(
                optimizer=Adam(learning_rate=0.0005),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            print("\nTraining in progress...")
            history = NN_Model_GA.fit(
                X_train_ann, y_train_np,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
                verbose=1
            )

            test_loss, test_acc = NN_Model_GA.evaluate(X_test_ann, y_test_np, verbose=0)
            print(f"\nANN Final Performance:")
            print(f"   Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
            print(f"   Test Loss: {test_loss:.4f}")

            predictions = NN_Model_GA.predict(X_test_ann, verbose=0)
            results_GA = (predictions > 0.5).astype(int)

            print(f"\n   Accuracy from GA prediction: {ann_accuracy:.4f}")
            print(f"   Actual accuracy after full training: {test_acc:.4f}")

            if test_acc >= ann_accuracy:
                print(f"    ANN trained successfully!")
            else:
                print(f"    Actual accuracy is lower than expected")

            NN_Model_GA.save('GA_ANN_model.h5')
            print("\nANN model saved as 'GA_ANN_model.h5'")

            np.save('ANN_predictions.npy', results_GA)
            print("Predictions saved as 'ANN_predictions.npy'")

    print(f"\n{'=' * 70}")
    print("SAVING SELECTED DATASETS FOR ALL MODELS")
    print('=' * 70)

    for model_name, result in results.items():
        features = result['best_features']

        if features:
            X_train_selected = X_train[features]
            X_test_selected = X_test[features]

            train_df = pd.DataFrame(X_train_selected, columns=features)
            train_df['target'] = y_train.values
            test_df = pd.DataFrame(X_test_selected, columns=features)
            test_df['target'] = y_test.values

            train_file = f"GA_{model_name}_train.csv"
            test_file = f"GA_{model_name}_test.csv"

            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)

            print(f"{model_name}: {len(features)} features -> {train_file}, {test_file}")

    print(f"\n{'=' * 70}")
    print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
    print('=' * 70)

else:
    print("\nNo models were successfully optimized")

print(f"\nSUMMARY OF BEST FEATURE SETS:")
for model_name in ['RandomForest', 'DecisionTree', 'ANN', 'KNN', 'SVM', 'LogisticRegression', 'NaiveBayes']:
    if model_name in results:
        features = results[model_name]['best_features']
        accuracy = results[model_name]['accuracy']
        print(f"\n{model_name}:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Features ({len(features)}): {features}")