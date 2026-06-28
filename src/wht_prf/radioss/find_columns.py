import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"dev_log\radioss_runs\cylinder_relax_law100\foam_relaxT01.csv", header=None)

# Col 0 is time
time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
time = np.nan_to_num(time)

plt.figure(figsize=(15, 10))

n_cols = df.shape[1]
plot_idx = 1
for i in range(1, n_cols):
    data = pd.to_numeric(df.iloc[:, i], errors='coerce').values
    data = np.nan_to_num(data)
    if np.ptp(data) > 1e-5:  # has some variation
        plt.subplot(6, 6, plot_idx)
        plt.plot(time, data)
        plt.title(f"Col {i}")
        plot_idx += 1
        if plot_idx > 36:
            break

plt.tight_layout()
plt.savefig("dev_log/radioss_runs/cylinder_relax_law100/columns_plot.png")
