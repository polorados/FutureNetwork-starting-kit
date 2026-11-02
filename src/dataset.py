import blosc2

def load_sample(folder, identifier):
    csi = blosc2.load_array(str(folder / f"CSI_{identifier}.b2"))
    ati = blosc2.load_array(str(folder / f"ATI_{identifier}.b2"))
    return csi, ati