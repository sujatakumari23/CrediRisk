# ============================
#  DAY 3: Model Training and Evaluation
# ============================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

@timer
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n🚀 Training Model: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("✔ Accuracy:", accuracy_score(y_test, y_pred))
    print("🎯 F1 Score:", f1_score(y_test, y_pred))
    print("🔥 ROC-AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

@timer
def main():
    FILE_PATH = r"C:\Users\HP\Downloads\train1.csv"
    TARGET = "credit_card_default"
    df = load_data(FILE_PATH)
    run_eda(df)
    df = feature_engineering(df)

    X, y, preprocessor = preprocess(df, TARGET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    models = get_models()

    plt.figure(figsize=(10, 6))
    for model_name, model_obj in models.items():
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('oversample', smote),
            ('classifier', model_obj)
        ])
        evaluate_model(model_name, pipeline, X_train, X_test, y_train, y_test)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Model ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    best_model = ImbPipeline([
        ('preprocessor', preprocessor),
        ('oversample', smote),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    best_model.fit(X_train, y_train)

if _name_ == '_main_':
    main()
