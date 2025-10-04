import pandas as pd
from ctgan import CTGAN
from utils import load_data




def main():
    df = load_data()
# Use only fraud examples to model minority distribution
    fraud = df[df['Class'] == 1].copy()
    fraud = fraud.drop('Class', axis=1)


    print('Fraud shape:', fraud.shape)


# Train CTGAN
    ctgan = CTGAN(epochs=300) # adjust epochs if needed
    ctgan.fit(fraud)


# Generate synthetic fraud samples
    synthetic = ctgan.sample(5000)
    synthetic['Class'] = 1


# Save synthetic
    synthetic.to_csv('data/synthetic_fraud.csv', index=False)


# Create augmented CSV: concat original + synthetic
    augmented = pd.concat([df, synthetic], ignore_index=True)
    augmented.to_csv('outputs/augmented_dataset.csv', index=False)
    print('Saved synthetic and augmented dataset.')


if __name__ == '__main__':
    main()