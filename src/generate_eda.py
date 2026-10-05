import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def generate_notebook(df):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Predictive Maintenance EDA\n",
                    "This notebook performs exploratory data analysis on the AI4I 2020 Predictive Maintenance Dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import os\n",
                    "\n",
                    "df = pd.read_csv('dataset/predictive_maintenance.csv')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f'Dimensions: {df.shape}')\n",
                    "print('\\nColumns:', df.columns.tolist())\n",
                    "print('\\nData Types:')\n",
                    "print(df.dtypes)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Missing Values:')\n",
                    "print(df.isnull().sum())\n",
                    "print('\\nDuplicate Records:', df.duplicated().sum())\n",
                    "print('\\nUnique Values per Column:')\n",
                    "print(df.nunique())\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Descriptive Statistics:')\n",
                    "display(df.describe())\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "target = 'Target'\n",
                    "print('Target Distribution:')\n",
                    "print(df[target].value_counts())\n",
                    "fails = df[target].sum()\n",
                    "norms = len(df) - fails\n",
                    "print(f'\\nNumber of machine failures: {fails}')\n",
                    "print(f'Number of normal records: {norms}')\n",
                    "print(f'Failure percentage: {(fails/len(df))*100:.2f}%')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'Failure Type' in df.columns:\n",
                    "    print('\\nAvailable Failure Categories:')\n",
                    "    print(df['Failure Type'].value_counts())\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "os.makedirs('results', exist_ok=True)\n",
                    "\n",
                    "def plot_hist(col, bins=30):\n",
                    "    safe_name = col.replace(' ', '_').replace('[', '').replace(']', '')\n",
                    "    plt.figure(figsize=(8,5))\n",
                    "    plt.hist(df[col], bins=bins, edgecolor='k')\n",
                    "    plt.title(f'{col} Distribution')\n",
                    "    plt.xlabel(col)\n",
                    "    plt.ylabel('Frequency')\n",
                    "    plt.savefig(f'results/{safe_name}_dist.png')\n",
                    "    plt.show()\n",
                    "\n",
                    "features_to_plot = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']\n",
                    "for feature in features_to_plot:\n",
                    "    if feature in df.columns:\n",
                    "        plot_hist(feature)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'Target' in df.columns:\n",
                    "    plt.figure(figsize=(6,4))\n",
                    "    df['Target'].value_counts().plot(kind='bar', color=['blue', 'red'])\n",
                    "    plt.title('Machine Failure Distribution')\n",
                    "    plt.ylabel('Count')\n",
                    "    plt.xticks(ticks=[0,1], labels=['Normal (0)', 'Failure (1)'], rotation=0)\n",
                    "    plt.savefig('results/Machine_failure_dist.png')\n",
                    "    plt.show()\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'Failure Type' in df.columns:\n",
                    "    plt.figure(figsize=(8,5))\n",
                    "    df['Failure Type'].value_counts().plot(kind='bar', color='orange')\n",
                    "    plt.title('Failure Type Distribution')\n",
                    "    plt.ylabel('Count')\n",
                    "    plt.xticks(rotation=45)\n",
                    "    plt.tight_layout()\n",
                    "    plt.savefig('results/Failure_type_dist.png')\n",
                    "    plt.show()\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
                    "corr = df[numeric_cols].corr()\n",
                    "plt.figure(figsize=(10,8))\n",
                    "plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)\n",
                    "plt.colorbar()\n",
                    "plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)\n",
                    "plt.yticks(range(len(corr.columns)), corr.columns)\n",
                    "plt.title('Correlation Heatmap')\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('results/Correlation_heatmap.png')\n",
                    "plt.show()\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open('notebooks/01_EDA.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

def main():
    csv_path = 'dataset/predictive_maintenance.csv'
    df = pd.read_csv(csv_path)
    
    print('--- DIMENSIONS ---')
    print(df.shape)
    
    print('\n--- COLUMNS ---')
    print(df.columns.tolist())
    
    print('\n--- DATA TYPES ---')
    print(df.dtypes)
    
    print('\n--- MISSING VALUES ---')
    print(df.isnull().sum())
    
    print('\n--- DUPLICATE RECORDS ---')
    print(df.duplicated().sum())
    
    print('\n--- UNIQUE VALUES ---')
    print(df.nunique())
    
    print('\n--- DESCRIPTIVE STATISTICS ---')
    print(df.describe())
    
    target = 'Target'
    print('\n--- TARGET DISTRIBUTION ---')
    print(df[target].value_counts())
    
    fails = df[target].sum()
    norms = len(df) - fails
    print(f'Number of machine failures: {fails}')
    print(f'Number of normal machine records: {norms}')
    print(f'Failure percentage: {(fails/len(df))*100:.2f}%')
    
    if 'Failure Type' in df.columns:
        print('\n--- AVAILABLE FAILURE CATEGORIES ---')
        print(df['Failure Type'].value_counts())
    else:
        print("None found.")
        
    print('\nGenerating visualizations...')
    os.makedirs('results', exist_ok=True)
    
    def plot_hist(col):
        # We replace spaces/brackets with underscores for filenames
        safe_name = col.replace(' ', '_').replace('[', '').replace(']', '')
        plt.figure(figsize=(8,5))
        plt.hist(df[col], bins=30, edgecolor='k')
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'results/{safe_name}_dist.png')
        plt.close()
        
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    for f in features:
        if f in df.columns:
            plot_hist(f)
            
    if target in df.columns:
        plt.figure(figsize=(6,4))
        df[target].value_counts().plot(kind='bar', color=['blue', 'red'])
        plt.title('Machine Failure Distribution')
        plt.ylabel('Count')
        plt.xticks(ticks=[0,1], labels=['Normal (0)', 'Failure (1)'], rotation=0)
        plt.savefig('results/Machine_failure_dist.png')
        plt.close()
        
    if 'Failure Type' in df.columns:
        plt.figure(figsize=(8,5))
        df['Failure Type'].value_counts().plot(kind='bar', color='orange')
        plt.title('Failure Type Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/Failure_type_dist.png')
        plt.close()
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('results/Correlation_heatmap.png')
    plt.close()
    
    print('Generating 01_EDA.ipynb...')
    generate_notebook(df)
    print('Done.')

if __name__ == '__main__':
    main()
