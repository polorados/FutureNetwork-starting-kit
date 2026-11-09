"""Load a compressed RoomSense sample and export an untrained PyTorch model.

Run from this folder with Python 3.14:
    python -m pip install -r requirements.txt
    python example.py /path/to/data/labeled --output model.onnx

Replace ExampleModel with your own trained model when preparing a submission.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import blosc2 # permet de lire les données compressées .b2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 5)

    def forward(self, frames, intervals):
        # frames de forme (600, 1, 3, 1024, 2) 600 snapshots dans le temps, 1 antenne, 3 canaux, 1024 sous-porteuses, 2 (réel, imaginaire)
        # on se retrouve à la fin de components avec un vecteur de taille 2 (réel et imaginaire) 
        # moyenné sur les 600 snapshots, 1 antenne, 3 canaux et 1024 sous-porteuses
        components = frames.abs().mean(dim=(1, 2, 3)).mean(dim=0)
        # calcule la moyennes des intervalles de temps entre les mesures 
        interval = intervals.mean().reshape(1)
        # on colle le components et l'interval pour obtenir un vecteur de taille 3 (réel_moy, imaginaire_moy, intervalle_moy)
        return self.classifier(torch.cat((components, interval)))


# Chaque échantillon à deux fichiers, CSI : mesures de signaux et ATI : intervalles de temps entre les mesures.
def load_sample(folder, identifier):
    csi = blosc2.load_array(str(folder / f"CSI_{identifier}.b2"))
    ati = blosc2.load_array(str(folder / f"ATI_{identifier}.b2"))
    return csi, ati


DATA_DIR_L = Path(__file__).parent / ".." / "data" / "labeled"
DATA_DIR_U = Path(__file__).parent / ".." / "data" / "unlabeled"
OUTPUT = "model.onnx"

def main():
    # ouvre le fichier samples.csv dans le dossier data_dir et lit la première ligne pour obtenir l'identifiant de l'échantillon à utiliser.
    with (DATA_DIR_L / "samples.csv").open(newline="") as stream:
        identifier = next(csv.DictReader(stream))["identifier"]
    csi, ati = load_sample(DATA_DIR_L, identifier)

    # Each complex64 value becomes two float32 values: real, then imaginary.
    frames = csi[:600].view(np.float32).reshape(600, 1, 3, 1024, 2)
    intervals = ati[:600].copy()
    print("dtype:", csi.dtype)
    print("Shape:", csi.shape)
    print("Min:", np.min(np.abs(csi)))
    print("Max:", np.max(np.abs(csi)))
    print("Non-zero:", np.count_nonzero(csi))
    print("First values:", csi[0, 0, 0, :10])


    intervals[-1] = 0 # on set la derniere intervalle à 0.
    # example est un tuple contenant les frames et les intervals convertis en tenseurs PyTorch.
    example = (torch.from_numpy(frames), torch.from_numpy(intervals))
    # .eval() transforme le modèle en mode prédiction. 
    model = ExampleModel().eval()
    # exporte le modèle pytorch en format onnx, en utilisant les frames et les intervales comme entrées. 
    torch.onnx.export(
        model, # modèle utilisé
        example, # input data
        OUTPUT,
        input_names=["frames", "intervals"], # le nom des inputs dans le modèle onnx
        output_names=["logits"], # le nom des outputs dans le modèle onnx
        opset_version=18, # version de l'opset. (Version des opérations supportées par ONNX)
        external_data=False, # sauvegarde les poids du modèle dans le fichier onnx au lieu de créer un fichier séparé. 
    )
    # Configure comment le modèle ONNX utilisera les threads du CPU pendant le calcul.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    # on load le fichier ONNX qu'on a exporté plus haut, en utilisant les options de threads définies précédemment.
    session = ort.InferenceSession(
        str(OUTPUT), sess_options=options, providers=["CPUExecutionProvider"]
    )
    # On met en input les frames et les intevales pour le modèle ONNX. nom des inputs "" définis plus haut. 
    inputs = {"frames": frames, "intervals": intervals}
    # On fait un premier run pour "chauffer" le modèle et éviter que le temps de prédiction soit faussé par la compilation initiale.
    # On définit l'ouput logits dans le cas où le modèle renvoie plusieurs outputs, mais on ne veut que les logits.
    session.run(["logits"], inputs)
    # on start un timer et on fait un run du modèle et on garde en mémoire les logits.
    start = perf_counter()
    logits = session.run(["logits"], inputs)[0]
    elapsed = perf_counter() - start
    # vérifie que la sortie du modèle ait la bonne forme et le bon type
    if logits.shape != (5,) or logits.dtype != np.float32:
        raise ValueError("Model output must be float32 with shape (5,)")
    # vérifie que le modèle ONNX ne dépasse pas 100 MB.
    size = args.output.stat().st_size
    if size >= 100_000_000:
        raise ValueError("The model must be smaller than 100 MB")
    print(f"Saved {args.output} ({size:,} bytes)") # : separe le nom de la variable et le formatteur. "," ajoute le formatage des milliers.
    # Donne la position du logits le plus élevé ce qui correspond au headcount.
    print(f"Predicted headcount: {logits.argmax()} (untrained example model)") 
    # Temps que ça a pris avec 3 nombres apres la decimale. 
    print(f"CPU prediction time: {elapsed:.3f} s/sample (after warm-up)")
    if elapsed >= 0.7:
        print(
            "Warning: prediction exceeds the 0.7 s/sample CPU budget. "
            "Estimate excludes loading overhead; server CPU speed may differ."
        )


if __name__ == "__main__":
    main()
