{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Nur-E-Anika/evolutionary-algorithm/blob/main/HO.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nt-8eVeu2026",
        "outputId": "f7dca37d-8283-41c2-aee4-593af23b9498"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dnnyNrHuywgq",
        "outputId": "0820345c-b303-4678-8e1c-a8237af850ff"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting opendatasets\n",
            "  Downloading opendatasets-0.1.22-py3-none-any.whl (15 kB)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from opendatasets) (4.66.1)\n",
            "Requirement already satisfied: kaggle in /usr/local/lib/python3.10/dist-packages (from opendatasets) (1.5.16)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from opendatasets) (8.1.7)\n",
            "Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.10/dist-packages (from kaggle->opendatasets) (1.16.0)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from kaggle->opendatasets) (2023.7.22)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from kaggle->opendatasets) (2.8.2)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from kaggle->opendatasets) (2.31.0)\n",
            "Requirement already satisfied: python-slugify in /usr/local/lib/python3.10/dist-packages (from kaggle->opendatasets) (8.0.1)\n",
            "Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from kaggle->opendatasets) (2.0.4)\n",
            "Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from kaggle->opendatasets) (6.0.0)\n",
            "Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->kaggle->opendatasets) (0.5.1)\n",
            "Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.10/dist-packages (from python-slugify->kaggle->opendatasets) (1.3)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle->opendatasets) (3.2.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle->opendatasets) (3.4)\n",
            "Installing collected packages: opendatasets\n",
            "Successfully installed opendatasets-0.1.22\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (1.5.3)\n",
            "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2023.3.post1)\n",
            "Requirement already satisfied: numpy>=1.21.0 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.23.5)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install opendatasets\n",
        "!pip install pandas"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RGU7-d-J52Ux",
        "outputId": "85a24e43-8730-4b2e-feca-9fa12d408fc0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Please provide your Kaggle credentials to download this dataset. Learn more: http://bit.ly/kaggle-creds\n",
            "Your Kaggle username: nureanikaanan\n",
            "Your Kaggle Key: ··········\n",
            "Downloading titanic.zip to ./titanic\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 34.1k/34.1k [00:00<00:00, 34.8MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Extracting archive ./titanic/titanic.zip to ./titanic\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ],
      "source": [
        "import opendatasets as od\n",
        "import pandas\n",
        "\n",
        "od.download(\"https://www.kaggle.com/competitions/titanic\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "IZAMU2fU6ipI"
      },
      "outputs": [],
      "source": [
        "# !pip install numpy==1.24.4\n",
        "# !pip install pandas --upgrade\n",
        "# !pip install matplotlib --upgrade\n",
        "\n",
        "import numpy as np # linear algebra\n",
        "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
        "import os"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install seaborn --upgrade\n",
        "!pip install sklearn --upgrade"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1axHCv4t7uwc",
        "outputId": "d9024e20-4bbd-4698-e986-a964e948d291"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: seaborn in /usr/local/lib/python3.10/dist-packages (0.13.0)\n",
            "Requirement already satisfied: numpy!=1.24.0,>=1.20 in /usr/local/lib/python3.10/dist-packages (from seaborn) (1.23.5)\n",
            "Requirement already satisfied: pandas>=1.2 in /usr/local/lib/python3.10/dist-packages (from seaborn) (1.5.3)\n",
            "Requirement already satisfied: matplotlib!=3.6.1,>=3.3 in /usr/local/lib/python3.10/dist-packages (from seaborn) (3.7.1)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (1.1.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (4.42.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (1.4.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (23.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (9.4.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (3.1.1)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.3->seaborn) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.2->seaborn) (2023.3.post1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.3->seaborn) (1.16.0)\n",
            "Requirement already satisfied: sklearn in /usr/local/lib/python3.10/dist-packages (0.0.post9)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "1tPjiFCO6WmA"
      },
      "outputs": [],
      "source": [
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import MinMaxScaler, OneHotEncoder\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "oIXDK5nw8thp"
      },
      "outputs": [],
      "source": [
        "# load dataset using pandas\n",
        "titanic_train_df = pd.read_csv('./titanic/train.csv')\n",
        "titanic_test_df = pd.read_csv('./titanic/test.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "2VDxi6DL9Ipn"
      },
      "outputs": [],
      "source": [
        "titanic_Y_col = titanic_train_df.columns[1]\n",
        "titanic_X_col = titanic_train_df.columns[2:]\n",
        "titanic_X_col = titanic_X_col.drop(['Name','Ticket'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "DyPiK5Cn9Zqn"
      },
      "outputs": [],
      "source": [
        "titanic_X, titanic_Y = titanic_train_df[titanic_X_col].copy(), titanic_train_df[titanic_Y_col].copy()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "tJkd_rj99lsX"
      },
      "outputs": [],
      "source": [
        "numeric_cols = titanic_train_df[titanic_X_col].select_dtypes(include=np.number).columns.tolist()\n",
        "categorical_cols = titanic_train_df[titanic_X_col].select_dtypes(exclude=np.number).columns.tolist()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "U_1TeKN79ryv"
      },
      "outputs": [],
      "source": [
        "# Impute and scale numeric columns\n",
        "imputer = SimpleImputer().fit(titanic_train_df[numeric_cols])\n",
        "titanic_X[numeric_cols] = imputer.transform(titanic_X[numeric_cols])\n",
        "\n",
        "\n",
        "scaler = MinMaxScaler().fit(titanic_X[numeric_cols])\n",
        "titanic_X[numeric_cols] = scaler.transform(titanic_X[numeric_cols])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o5-fQpSv9yLI",
        "outputId": "1f09ae50-f029-494a-947c-fafaacfd9891"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.\n",
            "  warnings.warn(\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n",
            "<ipython-input-17-e4e5141b2300>:4: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`\n",
            "  titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])\n"
          ]
        }
      ],
      "source": [
        "# One-hot encode categorical columns\n",
        "encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(titanic_X[categorical_cols])\n",
        "encoded_cols = list(encoder.get_feature_names_out(categorical_cols))\n",
        "titanic_X[encoded_cols] = encoder.transform(titanic_X[categorical_cols])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "661JrU5T-owI"
      },
      "outputs": [],
      "source": [
        "titanic_X = titanic_X[numeric_cols + encoded_cols]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "B9NEtRtI-t-A"
      },
      "outputs": [],
      "source": [
        "titanic_X_Train, titanic_X_Test, titanic_Y_Train, titanic_Y_Test = train_test_split(titanic_X, titanic_Y, test_size = 0.30,random_state = 42)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "p8pIPoNK-5dQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e7529966-56b0-423c-f1f7-d6bcf8c58250"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting pymoo\n",
            "  Downloading pymoo-0.6.0.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (2.5 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.5/2.5 MB\u001b[0m \u001b[31m12.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.10/dist-packages (from pymoo) (1.23.5)\n",
            "Requirement already satisfied: scipy>=1.1 in /usr/local/lib/python3.10/dist-packages (from pymoo) (1.11.2)\n",
            "Requirement already satisfied: matplotlib>=3 in /usr/local/lib/python3.10/dist-packages (from pymoo) (3.7.1)\n",
            "Requirement already satisfied: autograd>=1.4 in /usr/local/lib/python3.10/dist-packages (from pymoo) (1.6.2)\n",
            "Collecting cma==3.2.2 (from pymoo)\n",
            "  Downloading cma-3.2.2-py2.py3-none-any.whl (249 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m249.1/249.1 kB\u001b[0m \u001b[31m18.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting alive-progress (from pymoo)\n",
            "  Downloading alive_progress-3.1.4-py3-none-any.whl (75 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m75.9/75.9 kB\u001b[0m \u001b[31m10.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting dill (from pymoo)\n",
            "  Downloading dill-0.3.7-py3-none-any.whl (115 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m115.3/115.3 kB\u001b[0m \u001b[31m10.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting Deprecated (from pymoo)\n",
            "  Downloading Deprecated-1.2.14-py2.py3-none-any.whl (9.6 kB)\n",
            "Requirement already satisfied: future>=0.15.2 in /usr/local/lib/python3.10/dist-packages (from autograd>=1.4->pymoo) (0.18.3)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (1.1.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (4.42.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (1.4.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (23.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (9.4.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (3.1.1)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3->pymoo) (2.8.2)\n",
            "Collecting about-time==4.2.1 (from alive-progress->pymoo)\n",
            "  Downloading about_time-4.2.1-py3-none-any.whl (13 kB)\n",
            "Collecting grapheme==0.6.0 (from alive-progress->pymoo)\n",
            "  Downloading grapheme-0.6.0.tar.gz (207 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m25.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: wrapt<2,>=1.10 in /usr/local/lib/python3.10/dist-packages (from Deprecated->pymoo) (1.15.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3->pymoo) (1.16.0)\n",
            "Building wheels for collected packages: grapheme\n",
            "  Building wheel for grapheme (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for grapheme: filename=grapheme-0.6.0-py3-none-any.whl size=210079 sha256=05352d2c988b5c4dad135143e30f43d4c996a41fb03f8bd06a02179e82b03334\n",
            "  Stored in directory: /root/.cache/pip/wheels/01/e1/49/37e6bde9886439057450c494a79b0bef8bbe897a54aebfc757\n",
            "Successfully built grapheme\n",
            "Installing collected packages: grapheme, dill, Deprecated, cma, about-time, alive-progress, pymoo\n",
            "Successfully installed Deprecated-1.2.14 about-time-4.2.1 alive-progress-3.1.4 cma-3.2.2 dill-0.3.7 grapheme-0.6.0 pymoo-0.6.0.1\n"
          ]
        }
      ],
      "source": [
        "!pip install -U pymoo"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "jaFweg5d_Nai"
      },
      "outputs": [],
      "source": [
        "from pymoo.algorithms.moo.nsga3 import NSGA3\n",
        "from pymoo.factory import get_problem, get_reference_directions, get_sampling, get_crossover, get_mutation, get_termination\n",
        "from pymoo.operators.selection.rnd import RandomSelection\n",
        "from pymoo.operators.crossover.sbx import SBX\n",
        "from pymoo.operators.mutation.pm import PolynomialMutation\n",
        "from pymoo.termination.default import DefaultMultiObjectiveTermination\n",
        "from pymoo.core.problem import Problem\n",
        "from pymoo.optimize import minimize\n",
        "from pymoo.operators.sampling.lhs import LHS"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "JNElZIJdAwqk"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import classification_report,confusion_matrix\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "sns.set_style('darkgrid')\n",
        "%matplotlib inline\n",
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.model_selection import GridSearchCV"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "J3Tw5cORA7yx"
      },
      "outputs": [],
      "source": [
        "def test_params(**params):\n",
        "    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params).fit(titanic_X_Train, titanic_Y_Train)\n",
        "    train_accuracy_score = accuracy_score(titanic_Y_Train, model.predict(titanic_X_Train))\n",
        "    val_accuracy_score = accuracy_score(titanic_Y_Test, model.predict(titanic_X_Test))\n",
        "    return train_accuracy_score, val_accuracy_score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "u_sqyoAA_Y1Z"
      },
      "outputs": [],
      "source": [
        "# define the hyperparameter optimization problem\n",
        "class HyperparameterOptimizationProblem(Problem):\n",
        "\n",
        "    def __init__(self,level):\n",
        "        # define the lower and upper bounds of the hyperparameters\n",
        "        # n_estimators: number of trees in the forest (integer)\n",
        "        # max_depth: maximum depth of each tree (integer)\n",
        "        # max_features: maximum number of features (integer)\n",
        "        # min_samples_leaf: minimum number of samples required to be at a leaf node (integer)\n",
        "        self.level = level\n",
        "        self.var_ranges = [\n",
        "            [(10, 500), (2, 8), (2,30), (1,5), (0, 0.3)],\n",
        "            [(10, 600), (2, 12), (2,40), (1,9), (0,0.5)]\n",
        "        ]\n",
        "        xl = np.array([10, 2, 2, 1, 0])\n",
        "        xu = np.array([600, 12, 40, 9, 0.5])\n",
        "\n",
        "        # initialize the problem with 4 variables and 2 objectives\n",
        "        super().__init__(n_var = 5, n_obj = 3,\n",
        "                         xl=[rng[0] for rng in self.var_ranges[level]],\n",
        "                         xu=[rng[1] for rng in self.var_ranges[level]]\n",
        "            )\n",
        "\n",
        "    def _evaluate(self, x, out, *args, **kwargs):\n",
        "        # evaluate each solution (each row of x)\n",
        "        f = np.zeros((x.shape[0], self.n_obj))\n",
        "        for i in range(x.shape[0]):\n",
        "            # get the hyperparameters\n",
        "            n_estimators = int(x[i, 0])\n",
        "            max_depth = int(x[i, 1])\n",
        "            min_samples_split = int(x[i, 2])\n",
        "            min_samples_leaf = int(x[i, 3])\n",
        "            min_weight_fraction_leaf = int(x[i, 4])\n",
        "\n",
        "\n",
        "            # build and train the random forest model\n",
        "            model = RandomForestClassifier(n_estimators=n_estimators,\n",
        "                                           max_depth=max_depth,\n",
        "                                           min_samples_split=min_samples_split,\n",
        "                                           min_samples_leaf = min_samples_leaf,\n",
        "                                           min_weight_fraction_leaf = min_weight_fraction_leaf,\n",
        "                                           max_features=\"sqrt\",\n",
        "                                           random_state=42,\n",
        "                                           n_jobs = -1)\n",
        "            model.fit(titanic_X_Train, titanic_Y_Train)\n",
        "\n",
        "            # predict on the test set\n",
        "            y_pred = model.predict(titanic_X_Test)\n",
        "\n",
        "            # calculate the accuracy, f1 and ROC/AUC score as the objectives\n",
        "            f[i, 0] = -accuracy_score(titanic_Y_Test, y_pred) # negate because we want to maximize\n",
        "            f[i, 1] = -f1_score(titanic_Y_Test, y_pred) # negate because we want to maximize\n",
        "            f[i, 2] = -roc_auc_score(titanic_Y_Test, y_pred) # negate because we want to maximize\n",
        "\n",
        "\n",
        "        # assign the objectives to the output dictionary\n",
        "        out[\"F\"] = f"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2UtYt5uI_3aE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2684d510-1590-482e-f95d-487aecfb39a1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<timed exec>:8: DeprecationWarning: Call to deprecated function (or staticmethod) get_reference_directions. (Please use `from pymoo.util.ref_dirs import get_reference_directions`)\n",
            "<timed exec>:19: DeprecationWarning: Call to deprecated function (or staticmethod) get_reference_directions. (Please use `from pymoo.util.ref_dirs import get_reference_directions`)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "WARNING: pop_size=50 is less than the number of reference directions ref_dirs=91.\n",
            "This might cause unwanted behavior of the algorithm. \n",
            "Please make sure pop_size is equal or larger than the number of reference directions. \n",
            "==========================================================\n",
            "n_gen  |  n_eval  | n_nds  |      eps      |   indicator  \n",
            "==========================================================\n",
            "     1 |      100 |      1 |             - |             -\n",
            "     2 |      200 |      1 |  0.000000E+00 |             f\n",
            "     3 |      300 |      1 |  0.000000E+00 |             f\n",
            "     4 |      400 |      1 |  0.000000E+00 |             f\n",
            "     5 |      500 |      1 |  0.000000E+00 |             f\n",
            "     6 |      600 |      2 |  1.1618619413 |         ideal\n",
            "     7 |      700 |      2 |  0.000000E+00 |             f\n",
            "     8 |      800 |      2 |  1.0000000000 |         ideal\n"
          ]
        }
      ],
      "source": [
        "%%time\n",
        "# create an instance of the problem\n",
        "problem = HyperparameterOptimizationProblem(level=0)\n",
        "problem1 = HyperparameterOptimizationProblem(level=1)\n",
        "\n",
        "# create an instance of NSGA-III algorithm\n",
        "algorithm = NSGA3(\n",
        "    pop_size= 100,\n",
        "    ref_dirs=get_reference_directions(\"das-dennis\", 3, n_partitions=12),\n",
        "    # sampling=get_sampling(\"int_random\"),\n",
        "    sampling=LHS(),\n",
        "    selection = RandomSelection(),\n",
        "    # crossover=get_crossover(\"int_sbx\", prob=0.9, eta=15),\n",
        "    crossover = SBX(prob=0.6, prob_var=0.5),\n",
        "    mutation=PolynomialMutation(prob=0.5),\n",
        "    eliminate_duplicates=True)\n",
        "\n",
        "algorithm1 = NSGA3(\n",
        "    pop_size= 50,\n",
        "    ref_dirs=get_reference_directions(\"das-dennis\", 3, n_partitions=12),\n",
        "    # sampling=get_sampling(\"int_random\"),\n",
        "    sampling=LHS(),\n",
        "    selection = RandomSelection(),\n",
        "    # crossover=get_crossover(\"int_sbx\", prob=0.9, eta=15),\n",
        "    crossover = SBX(prob=0.9, prob_var=0.8),\n",
        "    mutation=PolynomialMutation(prob=0.8),\n",
        "    eliminate_duplicates=True)\n",
        "\n",
        "# create an instance of termination criterion\n",
        "# termination = get_termination(\"n_gen\", 50)\n",
        "\n",
        "# early stop\n",
        "termination = DefaultMultiObjectiveTermination(\n",
        "    xtol=1e-8,           # movement in the design space xtol\n",
        "    cvtol=1e-6,          # the convergence in the constraint cv_tol\n",
        "    ftol=0.0025,         # objective space f_tol.\n",
        "    period=30,\n",
        "    n_max_gen=50,        # maximum number of generations n_max_gen\n",
        "    n_max_evals=100000   # function evaluations n_max_evals\n",
        ")\n",
        "\n",
        "# perform the optimization\n",
        "res = minimize(problem,\n",
        "               algorithm,\n",
        "               termination,\n",
        "               seed=42,\n",
        "               save_history=True,\n",
        "               verbose=True)\n",
        "res1 = minimize(problem1,\n",
        "               algorithm1,\n",
        "               termination,\n",
        "               seed=42,\n",
        "               seed_population=res.pop,\n",
        "               save_history=True,\n",
        "               verbose=True)\n",
        "\n",
        "# print the results\n",
        "print(f\"Best solution found: \\nX = {res1.X.astype(int)} \\nF = {-res1.F}\") # negate F because we maximized\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "NlFRxFda7ICf"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}