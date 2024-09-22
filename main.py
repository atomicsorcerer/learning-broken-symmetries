from matplotlib import pyplot as plt
import polars as pl

general_output = pl.read_csv("classifiers/general/analysis.csv").select(["muon_inv_mass", "output", "pT"])
invariant_output = pl.read_csv("classifiers/invariant/analysis.csv").select(["muon_inv_mass", "output", "pT"])
hybrid_output = pl.read_csv("classifiers/hybrid/pT_vs_acc_analysis.csv").select(["muon_inv_mass", "output", "pT"])

figure, axis = plt.subplots(3, 3, sharex=True, sharey=True)

figure.suptitle("Comparison of mass vs. output for different slices of pT (blur=10%)")

output_titles = ["General PFN Classifier", "Invariant Classifier", "Hybrid Classifier"]
outputs = [general_output, invariant_output, hybrid_output]
colors = ["tab:blue", "tab:orange", "tab:green"]
intervals = [0, 30, 40, 300]
for i in range(3):
	for j in range(3):
		axis[i, j].scatter(
			outputs[j].filter(pl.col("pT").is_between(intervals[i], intervals[i + 1])).to_numpy()[..., 0],
			outputs[j].filter(pl.col("pT").is_between(intervals[i], intervals[i + 1])).to_numpy()[..., 1],
			c=colors[j], s=0.1, marker=".")
		axis[i, j].set_title(f"{output_titles[j]} ({intervals[i]} < pT < {intervals[i + 1]})")

figure.set_size_inches(12, 8)
plt.savefig('figures/mass vs output.pdf', dpi=600)
plt.show()
