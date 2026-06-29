import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_balanced_model():
    # 1. Load Data
    df = pd.read_csv(r"D:\PES\2nd sem\aiml\project ai studio\fully_cleaned_student_pathway_data.csv") # Use your filename here

    # 2. Preprocessing
    df = df.drop('RollNo', axis=1)
    
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    le_stream = LabelEncoder()
    df['Recommended_Stream'] = le_stream.fit_transform(df['Recommended_Stream'])

    X = df.drop('Recommended_Stream', axis=1)
    y = df['Recommended_Stream']

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. BALANCED MODEL
    # class_weight='balanced' forces the model to treat Academic as equally important to Technical
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=12, 
        class_weight='balanced', 
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # 5. Verify Results
    accuracy = model.score(X_test, y_test)
    print(f" Balanced Model Trained. Accuracy: {accuracy * 100:.2f}%")
    
    # Show distribution check
    preds = model.predict(X_test)
    print("Prediction counts on test set:")
    print(pd.Series(le_stream.inverse_transform(preds)).value_counts())

    # 6. Save
    joblib.dump(model, 'stream_model.pkl')
    joblib.dump(le_gender, 'gender_encoder.pkl')
    joblib.dump(le_stream, 'stream_encoder.pkl')

if __name__ == "__main__":
    train_balanced_model()