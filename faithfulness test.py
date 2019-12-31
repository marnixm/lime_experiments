import matplotlib.pyplot as plt
import numpy as np
model, explain, faith = [],[],[]
model.extend([0,0.07])
explain.extend([0,0.073])

for i in range(10):
    model.append(0.073)
    explain.append(0.073+(i+1)*0.001)
    faith.append(np.corrcoef(model, explain)[0, 1])

print("model ", model)
print("explain", explain)
print()
print(faith)
plt.scatter(model, explain)
plt.show()
# the correlation is nearly 1. Even though the explainer is pointing at features with very small impact, which the model is not using
# but, is that undesirable behaviour? Th explainer is 'forced' to find 10 features, but it does only give them a tiny importance.

