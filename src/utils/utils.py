import scipy.io
import scipy.sparse
import pandas as pd
import pickle as pkl
def mtx_to_npz(mtx_paths:list,npz_path:str):
    def load_mtx_file(mtx_file):
        matrix = scipy.io.mmread(mtx_file)
        if not isinstance(matrix, scipy.sparse.csr_matrix):
            matrix = matrix.tocsr()
        return matrix

    # Load the first matrix to start the concatenation
    concatenated_matrix = load_mtx_file(mtx_paths[0])

    # Incrementally concatenate the remaining matrices
    for mtx_file in mtx_paths[1:]:
        next_matrix = load_mtx_file(mtx_file)
        concatenated_matrix = scipy.sparse.hstack([concatenated_matrix, next_matrix])

    # Save the concatenated matrix to an NPZ file
    scipy.sparse.save_npz(npz_path, concatenated_matrix)
    print(f"Concatenated matrix saved to {npz_path} of shape {concatenated_matrix.shape}")

def get_subsetted_genes(data_df:pd.DataFrame,celltype_to_genenames: dict)-> dict:
    subsetted = {}
    if celltype_to_genenames:
        for celltype,genenames in celltype_to_genenames.items():
            subsetted[celltype] = data_df.loc[:,genenames]
    return subsetted

def save_model(model,path = "xgb_reg.pkl"):
    """
    Adds an option to save model
    """
    
    pkl.dump(model, open(path, "wb"))
    return