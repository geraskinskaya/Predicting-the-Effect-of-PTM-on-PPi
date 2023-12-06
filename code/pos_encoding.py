def positional_encoding(seq_len, d_model):
    # Calculate positional encoding for a specific sequence length and model depth.
    # seq_len: length of sequences
    # d_model: depth of model (i.e., dimension of word embeddings)

    # Initialize a matrix with zeros of shape (seq_len, d_model)
    pos_enc = np.zeros((seq_len, d_model))

    # Calculate positional encoding
    for pos in range(seq_len):
        for i in range(0, d_model, 2):  # iterate over even indices up to d_model
            # Apply sine to even indices
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:  # check if 'i' is not the last dimension
                # Apply cosine to odd indices
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

    return pos_enc