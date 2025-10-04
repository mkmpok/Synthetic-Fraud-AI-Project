import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    orig = pd.read_csv('data/creditcard.csv')
    synth = pd.read_csv('data/synthetic_fraud.csv')
    # For visualization, sample some real fraud rows
    real_fraud = orig[orig['Class']==1].drop('Class', axis=1)
    real_sample = real_fraud.sample(min(500, len(real_fraud)), random_state=42)
    synth_sample = synth.drop('Class', axis=1).sample(min(500, len(synth)), random_state=42)
    combined = pd.concat([real_sample, synth_sample], ignore_index=True)
    labels = [0]*len(real_sample) + [1]*len(synth_sample)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embedded = tsne.fit_transform(combined)
    plt.scatter(embedded[:len(real_sample),0], embedded[:len(real_sample),1], label='real', alpha=0.6)
    plt.scatter(embedded[len(real_sample):,0], embedded[len(real_sample):,1], label='synthetic', alpha=0.6)
    plt.legend()
    plt.title('t-SNE: real vs synthetic fraud')
    plt.savefig('outputs/tsne_real_vs_synth.png')
    # PCA
    pca = PCA(n_components=2)
    p = pca.fit_transform(combined)
    plt.clf()
    plt.scatter(p[:len(real_sample),0], p[:len(real_sample),1], label='real', alpha=0.6)
    plt.scatter(p[len(real_sample):,0], p[len(real_sample):,1], label='synthetic', alpha=0.6)
    plt.legend()
    plt.title('PCA: real vs synthetic fraud')
    plt.savefig('outputs/pca_real_vs_synth.png')
    print('Saved t-SNE and PCA plots in outputs/')
if __name__ == '__main__':
    main()