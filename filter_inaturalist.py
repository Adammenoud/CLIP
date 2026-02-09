import pandas as pd
import os
from tqdm import tqdm


def missing_per_column(cols, file_path, chunk_size=1_000_000):
    """
    Count missing values per column in a large CSV file chunk-wise.
    
    Args:
        cols (list): List of column names to analyze.
        file_path (str): Path to the CSV file (tab-separated).
        chunk_size (int): Number of rows per chunk.
        
    Returns:
        pd.Series: Number of missing values per column.
    """
    na_counts = None

    # Read file in chunks
    for chunk in pd.read_csv(file_path, sep='\t', usecols=cols, chunksize=chunk_size):
        chunk_na = chunk.isna().sum()  # missing values per column in this chunk
        if na_counts is None:
            na_counts = chunk_na
        else:
            na_counts = na_counts.add(chunk_na, fill_value=0)
    
    # Convert counts to integers
    na_counts = na_counts.astype(int)
    print(na_counts)



def get_filtered_copy(file_path, callback_filter, chunksize=1_000_000,sep="\t", usecols=None, output_path=None,**kwargs):
    '''Go through a file by chunks and appies the filter on all the chunks, and rewrites the file accordingly.
    callback_filter must take a pandas dataframe as input and return also a panda dataframe (filtered/modified.)
    Careful: the output_path argument leads to an existing file, it will detlete it.
    '''
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_filtered{ext}"
    if os.path.exists(output_path):
        os.remove(output_path)
    if usecols ==None: #If none, take all columns
        usecols=pd.read_csv(file_path,sep=sep,nrows=0).columns
    write_header = True #write only once
    counter=0

    for chunk in pd.read_csv(
        file_path,
        sep=sep,
        usecols=usecols,
        chunksize=chunksize,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    ):
        chunk=callback_filter(chunk, **kwargs)
        chunk.to_csv(
            output_path,
            mode="a",
            header=write_header,
            index=False
        )
        write_header = False #turns it off
        print(f"chunk {counter} processed.")
        counter+=1

def filter_France_and_taxa(chunk, mushrooms):
        """
        Keeps only the rows in FR that are not na, as well as only mushrooms or arthropods
        """
        chunk = chunk.dropna()
        chunk = chunk[chunk['countryCode']=="FR"]
        if mushrooms:
            chunk = chunk[chunk['kingdom'] == 'Fungi']
        else:
            chunk = chunk[(chunk['phylum'] == 'Arthropoda')]
        return chunk

def filter_France_and_plants(chunk):
    chunk = chunk.dropna()
    chunk = chunk[chunk['countryCode']=="FR"]
    chunk = chunk[chunk['kingdom'] == 'Plantae']
    return chunk
def equalize_dataframes(file_to_filter, reference_file, column ,sep="\t", output_dir=None,chunksize=1_000_000):
    '''
    Creates a copy of a large file (without loading into memory), from a reference file (that should fit in memory).
    Keeps only the row for which "column" value also exists in the reference.
    '''

    ref=pd.read_csv(reference_file)
    ref_values = set(ref[column].dropna().astype(str))
    print(f"there is {len(ref_values)} different gbifID")
    if output_dir is None:
        base, ext = os.path.splitext(file_to_filter)
        output_dir = f"{base}_filtered{ext}"
    if os.path.exists(output_dir):
        os.remove(output_dir)

    write_header = True #write only once

    for chunk in tqdm(pd.read_csv(
        file_to_filter,
        sep=sep,
        chunksize=chunksize,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
        )):
        #filter
        chunk = chunk[chunk[column].isin(ref_values)]
        #write
        chunk.to_csv(
            output_dir,
            mode="a",
            header=write_header,
            index=False
        )
        write_header = False


if __name__ == "__main__": 
    
    '''
    We create filtered versions of the inaturalist occurence and multimedia files, keeping only the rows with non NA values in the columns of interest, and only for France.
    We create one version for arthropods, one for mushrooms and one for plants.

    The corresponding multimedia files are created by keeping only the rows with a gbifID present in the occurence files.

    One occurence file can have several corresponding multimedia rows, since one occurence can have several pictures. 
    This problem is addressed by the 
    '''
    occ_path="Data/data_inaturalist/occurrence.txt"   #raw occurence file, should not be loaded in memory, we create filtered copies of it.
    mult_path="Data/data_inaturalist/multimedia.txt"   #likewise, mutlimedia file. Contains the links to download the images.
    taxa_cols= ['kingdom','phylum','class', 'order','family','genus','species']
    columns= taxa_cols + ['gbifID','countryCode',"scientificName", "decimalLatitude", "decimalLongitude"]

    #Check the NA by batches, since the whole file cannot be loaded in memory.
    missing_per_column(columns, occ_path)
    #Results show that the missing values are negligeable; we keep only the rows for which we have all the needed information.


    #Create the occurence files we need         
    print("filter mushrooms")
    get_filtered_copy(occ_path, filter_France_and_taxa, chunksize=1_000_000,sep="\t", usecols=columns, output_path="Data/filtered_inaturalist/occurrence_mushrooms.txt",mushrooms=True)
    print("filter arthropods")
    get_filtered_copy(occ_path, filter_France_and_taxa, chunksize=1_000_000,sep="\t", usecols=columns, output_path="Data/filtered_inaturalist/occurrence_arthropods.txt",mushrooms=False)
    print("filter plants")
    get_filtered_copy(occ_path, filter_France_and_plants, chunksize=1_000_000,sep="\t", usecols=columns, output_path="Data/filtered_inaturalist/occurrence_plants.txt")

    #Create the corresponding mutlimedia files
    equalize_dataframes(file_to_filter=mult_path, 
                        reference_file="Data/filtered_inaturalist/occurrence_arthropods.txt",
                        column='gbifID',
                        output_dir="Data/filtered_inaturalist/multimedia_arthropods.txt"
                        )
    equalize_dataframes(file_to_filter=mult_path, 
                        reference_file="Data/filtered_inaturalist/occurrence_mushrooms.txt",
                        column='gbifID',
                        output_dir="Data/filtered_inaturalist/multimedia_mushrooms.txt"
                        )
    equalize_dataframes(file_to_filter=mult_path, 
                        reference_file="Data/filtered_inaturalist/occurrence_plants.txt",
                        column='gbifID',
                        output_dir="Data/filtered_inaturalist/multimedia_plants.txt"
                        )