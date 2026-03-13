
import pandas as pd
import matplotlib.pyplot as plt

def get_fct(colx,colf,df,mean=False):
    if mean:
        return df.groupby(colx)[colf].mean()
    else:
        return df.groupby(colx)[colf]
'''def quick_plot(df,colx, colf="auc_emb_PR",kind="bar",center=True,title=None, mean=False):
    if title is None:
        title=f'Mean {colf} per {colx}'
    mean_values=get_fct(colx,colf,df,mean)
    mean_values.plot(kind=kind)  
    plt.ylabel(f'Mean {colf}')
    plt.xlabel(f'{colx}')
    plt.title(title)
    
    if kind=="bar" and center==True:
        ymin = mean_values.min() - 0.01
        ymax = mean_values.max() + 0.01
        plt.ylim(ymin, ymax)
    plt.show()'''

def quick_plot(
    df,
    colx,
    colf="auc_emb_PR",
    kind="bar",
    center=True,
    title=None,
    mean=False,
    xlabel=None,
    ylabel=None,
    rename_index=None
):
    '''
    xlabel : str or None, default=None
        Custom label for the x-axis. If None, `colx` is used.

    ylabel : str or None, default=None
        Custom label for the y-axis. If None, a default label based on `colf`
        is used.

    rename_index : dict or None, default=None
        Optional dictionary mapping original group labels to new labels.
        Useful for renaming categories displayed on the x-axis.
         e.g. : {"bs": "Batch size",
                "lr": "Learning rate" }
    '''


    if title is None:
        title = f"{colf} per {colx}"

    values = get_fct(colx, colf, df, mean)

    # Rename x categories if mapping provided
    if rename_index is not None:
        values = values.rename(index=rename_index)

    values.plot(kind=kind)

    # Custom axis labels
    plt.xlabel(xlabel if xlabel else colx)
    plt.ylabel(ylabel if ylabel else f"{colf}")
    plt.title(title)

    if kind == "bar" and center:
        ymin = values.min() - 0.01
        ymax = values.max() + 0.01
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
