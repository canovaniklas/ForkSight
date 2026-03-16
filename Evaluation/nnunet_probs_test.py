from pathlib import Path
import numpy as np

p = Path("C:\\Users\\juhe9\\Downloads\\nnUNetTrainerWandb__nnUNetPlans__2d\\best_configuration_inference_output\\forksight_0395.npz")
data = np.load(p)
print(data)
probs = data["probabilities"]
print(probs.shape)
print(np.min(probs[0]), np.max(probs[0]))
print(np.min(probs[1]), np.max(probs[1]))

