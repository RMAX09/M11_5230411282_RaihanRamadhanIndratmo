import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def input_data():
    """
    Input and preprocess the Excel dataset with robust file handling
    """
    while True:
        try:
            # Prompt for file name
            file_name = input(
                "Masukkan nama file Excel (contoh: data.xlsx): ").strip()

            # Check file existence and extension
            if not file_name.endswith('.xlsx'):
                print("Error: Hanya file Excel (.xlsx) yang didukung.")
                continue

            if not os.path.exists(file_name):
                print(f"Error: File {file_name} tidak ditemukan.")
                continue

            # Read Excel file
            df = pd.read_excel(file_name)

            # Check if dataframe is empty
            if df.empty:
                print("Error: File kosong atau tidak dapat dibaca.")
                continue

            print("\nData Awal:")
            print(df.head())

            # Check and print null values
            print("\nNilai Kosong:")
            print(df.isnull().sum())

            # Identify numerical and categorical columns
            numerical_columns = df.select_dtypes(
                include=['float64', 'int64']).columns.tolist()
            categorical_columns = df.select_dtypes(
                include=['object']).columns.tolist()

            # Fill null values
            for col in numerical_columns:
                df[col].fillna(df[col].median(), inplace=True)

            # Handle categorical columns (if any exist)
            for col in categorical_columns:
                # Fill nulls with mode
                df[col].fillna(df[col].mode()[0], inplace=True)

                # Label Encoding
                label_enc = LabelEncoder()
                df[col] = label_enc.fit_transform(df[col])

            # Scaling numerical features
            scaler = MinMaxScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

            return df, categorical_columns, numerical_columns

        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
            print("Silakan coba lagi.")


def run_algorithm(df, categorical_columns, numerical_columns):
    """
    Run machine learning algorithms on the dataset
    """
    # Determine the target column (assumed to be the first categorical column)
    target_column = categorical_columns[0] if categorical_columns else None

    if target_column is None:
        print("Error: Tidak dapat menemukan kolom target.")
        return

    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("\nJumlah data train:", len(X_train))
    print("Jumlah data test:", len(X_test))

    # Algorithm selection menu
    print("\nPilih algoritma untuk analisis:")
    print("1. K-Nearest Neighbors (KNN)")
    print("2. Random Forest")
    print("3. Naive Bayes")
    choice = input("Masukkan pilihan (1/2/3): ")

    # Select and train model
    if choice == '1':
        print("\n== K-Nearest Neighbors ==")
        model = KNeighborsClassifier(n_neighbors=5)
    elif choice == '2':
        print("\n== Random Forest ==")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif choice == '3':
        print("\n== Naive Bayes ==")
        model = GaussianNB()
    else:
        print("Pilihan tidak valid.")
        return

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics
    print("\nAkurasi:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    print("===== Program Analisis Dataset =====")
    print("Selamat datang!")

    df, categorical_columns, numerical_columns = None, None, None
    while True:
        print("\nMenu:")
        print("1. Input Data")
        print("2. Pilih Algoritma dan Analisis Data")
        print("3. Keluar")
        menu_choice = input("Pilih menu (1/2/3): ")

        if menu_choice == '1':
            df, categorical_columns, numerical_columns = input_data()
            print("\nData berhasil diproses!")

        elif menu_choice == '2':
            if df is not None:
                run_algorithm(df, categorical_columns, numerical_columns)
            else:
                print("Anda harus input data terlebih dahulu (menu 1).")

        elif menu_choice == '3':
            print("Terima kasih! Program selesai.")
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")


if __name__ == "__main__":
    main()
