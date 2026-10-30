from Bio.Data.IUPACData import protein_letters_3to1_extended
THREE_TO_ONE = {k.upper(): v.upper() for k, v in protein_letters_3to1_extended.items()}
THREE_TO_ONE.update({'MSE': 'M', 'UNK': 'X'})
