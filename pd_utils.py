
import pandas as pd
import matplotlib.pyplot as plt

def get_fct(colx,colf,df):
    return df.groupby(colx)[colf].mean()
def quick_plot(df,colx, colf="auc_emb_PR",kind="bar",center=True,title=None):
    if title is None:
        title=f'Mean {colf} per {colx}'
    mean_values=get_fct(colx,colf,df)
    mean_values.plot(kind=kind)  
    plt.ylabel(f'Mean {colf}')
    plt.xlabel(f'{colx}')
    plt.title(title)
    
    if kind=="bar" and center==True:
        ymin = mean_values.min() - 0.01
        ymax = mean_values.max() + 0.01
        plt.ylim(ymin, ymax)
    plt.show()


#highest:
def get_highest(df):
    best_row_PR = df.loc[df['auc_emb_PR'].idxmax()]
    best_row_MLP = df.loc[df['auc_emb_MLP'].idxmax()]
    print("best PR:",best_row_PR["auc_emb_PR"],best_row_PR["run_name"])
    print("best MLP:",best_row_MLP["auc_emb_PR"],best_row_PR["run_name"])

def quick_filter(df,show_alpha_earth=False, **kwargs):
    filtered_df = df

    for col, val in kwargs.items():
        if val is None:
            continue
        # Treat scalars as a list of one element
        if isinstance(val, (list, tuple, set)):
            values = val
        else:
            values = [val]
        filtered_df = filtered_df[filtered_df[col].isin(values)]
    if show_alpha_earth:
        show_alpha_earth = df.loc[df["run_name"] == "alpha_earth"]
        show_alpha_earth = show_alpha_earth.fillna("alpha_earth")
        filtered_df = pd.concat([filtered_df, show_alpha_earth], ignore_index=True)
    return filtered_df