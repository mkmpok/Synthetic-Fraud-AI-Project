import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

def main():
    df = load_data()
    print('Shape:', df.shape)
    print(df['Class'].value_counts())


# Class distribution
    sns.countplot(x='Class', data=df)
    plt.title('Class distribution')
    plt.savefig('outputs/class_distribution.png')
    plt.clf()

# Correlation heatmap (small sample to speed up)
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', vmax=0.6)
    plt.title('Feature correlation')
    plt.savefig('outputs/corr_heatmap.png')

if __name__ == '__main__':
    main()