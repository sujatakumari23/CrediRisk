@timer
def run_eda(df):
    describe_data(df)
    print("\n📊 Missing values:\n", df.isnull().sum())
    print("\n📈 Class Distribution:\n", df[df.columns[-1]].value_counts())

    plt.figure(figsize=(8, 4))
    sns.countplot(x=df.columns[-1], data=df)
    plt.title("Target Variable Distribution")
    plt.show()

    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_features) > 1:
        plt.figure(figsize=(14, 6))
        sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

    for col in numerical_features:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()


@timer
def feature_engineering(df):
    df = df.copy()
    if 'Income' in df.columns and 'LoanAmount' in df.columns:
        df['Income_to_Loan_Ratio'] = df['Income'] / (df['LoanAmount'] + 1)
    if 'Age' in df.columns:
        df['Age_Bin'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], labels=["Young", "Mid-Age", "Senior", "Elder"])
    return df

@timer
def preprocess(df, target):
    df = df.dropna(subset=[target]) # Drop rows with NaN in the target column before splitting.
    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    return X, y, preprocessor
