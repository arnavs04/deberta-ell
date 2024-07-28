import numpy as np
import pandas as pd
import tqdm
import sklearn
import torch
import transformers
import tokenizers
import ptflops
import iterstrat

print(f"numpy=={np.__version__}")
print(f"pandas=={pd.__version__}")
print(f"tqdm=={tqdm.__version__}")
print(f"scikit-learn=={sklearn.__version__}")
print(f"torch version=={torch.__version__}")
print(f"iterstrat=={iterstrat.__version__}")
print(f"transformers=={transformers.__version__}")
print(f"tokenizers=={tokenizers.__version__}")
print(f"ptflops=={ptflops.__version__}")