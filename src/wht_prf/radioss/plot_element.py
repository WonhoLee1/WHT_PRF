import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"dev_log\radioss_runs\cylinder_relax_law100\foam_relaxT01.csv", header=None)

# Force numeric
time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
time = np.nan_to_num(time)

plt.figure(figsize=(12, 8))

# Let's plot Col 44 to Col 55 to see what they are
plot_idx = 1
for i in range(44, 56):
    data = pd.to_numeric(df.iloc[:, i], errors='coerce').values
    data = np.nan_to_num(data)
    plt.subplot(3, 4, plot_idx)
    plt.plot(time, data)
    plt.title(f"Col {i}")
    plot_idx += 1

plt.tight_layout()
plt.savefig("dev_log/radioss_runs/cylinder_relax_law100/element_vars.png")
