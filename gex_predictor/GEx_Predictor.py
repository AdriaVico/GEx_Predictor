import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize  import rdMolStandardize
from signaturizer import Signaturizer
import sys 
from pathlib import Path


project_root = Path(__file__).parent.parent.resolve()  
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from model.models import GenomicExpressionNet2

# Ensure the model directory is in the Python path
model_dir = project_root / "model"
if str(model_dir) not in sys.path:
    sys.path.append(str(model_dir))

class GEx_Predictor():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        project_root = Path(__file__).parent.parent.resolve()
        pt_path = project_root / "model" / "fold_4.pt"
        print(pt_path)
        print(f"[INFO] Loading trained model from {pt_path} ...")
        self.model = torch.load(pt_path, map_location=self.device, weights_only=False)
        print("[INFO] Model loaded successfully.")

    def standardize_smiles(self, input_smiles):
        standardize = []
        for smile in input_smiles:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    return None

                mol = rdMolStandardize.Cleanup(mol)

                mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
                mol = rdMolStandardize.Uncharger().uncharge(mol)

                # Add tautomer canonicalization if needed
                mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)

                Chem.SanitizeMol(mol)

                standardize.append(Chem.MolToSmiles(mol))

            except Exception as e:
                print(f"Could not process {input_smiles}: {e}")
                return None

        return standardize

    def get_GLOBAL_Signature(self, input_smiles):

        # We parse our standarized smiles to GLOBAL Signature
        signature = 'GLOBAL'
        sign = Signaturizer(signature)
        results = sign.predict(input_smiles)

        return results.signature[:]
    

    def predict(self, input_smile, input_type = "SMILES"):

        print(f"[INFO] Starting prediction for input type: {input_type}")
        print("INPUT", input_smile.shape)
        if input_type.upper() == "SMILES":
            standardized_smiles = self.standardize_smiles(input_smile)
            if standardized_smiles is None:
                raise ValueError("[ERROR] Failed to standardize input SMILES.")
        else:
            raise NotImplementedError("Only 'SMILES' input_type is implemented.")

        signatures_to_predict = self.get_GLOBAL_Signature(standardized_smiles)

        print("[STEP] Converting signatures to tensor and performing prediction...")
        x = torch.tensor(signatures_to_predict, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(x).cpu().numpy()

        print("[INFO] Prediction completed successfully.")


        return preds