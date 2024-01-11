import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_metrics(metrics_df):
    """
    Plot the metrics contained in metrics_df
    :param metrics_df: a pandas dataframe (columns: 'clf', 'metric', 'score')
    """
    ax = sns.barplot(data=metrics_df, y='clf', x='score', hue='metric')
    ax.set(xlabel='Score', ylabel='Classifier')
    ax.set(xlim=(0, 1.09))
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    nb_clf = len(metrics_df['clf'].unique())
    if nb_clf > 5:
        plt.rc('font', size=8)

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f')

    plt.rc('font', size=12)
    plt.show()

def plot_time(time_df):
    """
    Plot the time contained in time_df
    :param time_df: a pandas dataframe (columns: 'clf', 'time')
    """
    ax = sns.barplot(data=time_df, y='clf', x='time')
    ax.set(xlabel='Time (s)', ylabel='Classifier')

    # Set padding inside the box
    max_time = time_df['time'].max()
    ax.set(xlim=(0, max_time + 0.18 * max_time))

    for bars in ax.containers:
        ax.bar_label(bars, fmt=' %.3f')
    
    plt.show()

if __name__ == '__main__':
    metrics_df = pd.DataFrame({'clf': ['clf1', 'clf1', 'clf1', 'clf2', 'clf2', 'clf2'],
                                 'metric': ['accuracy', 'precision', 'recall', 'accuracy', 'precision', 'recall'],
                                 'score': [0.8, 0.9, 0.7, 0.7, 0.8, 0.6]})
    plot_metrics(metrics_df)

    time_df = pd.DataFrame({'clf': ['clf1', 'clf2'],
                            'time': [0.1, 0.2]})
    plot_time(time_df)