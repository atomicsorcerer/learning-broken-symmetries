from matplotlib import pyplot as plt
import polars as pl

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 1.5})
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': "serif"})
plt.rcParams.update({'font.serif': "Computer Modern Serif"})

# AUC vs. epoch

n_ensembles = 5

general_log = pl.read_csv("classifiers/general/training logs/log_*.csv").select(
	["auc"]).to_numpy().reshape(n_ensembles, -1).transpose()
invariant_log = pl.read_csv("classifiers/invariant/training logs/log_*.csv").select(
	["auc"]).to_numpy().reshape(n_ensembles, -1).transpose()
hybrid_log = pl.read_csv("classifiers/hybrid/training logs/log_*.csv").select(
	["auc"]).to_numpy().reshape(n_ensembles, -1).transpose()

x_axis = [i for i in range(1, 101)]

figure = plt.figure()
figure.set_size_inches(12, 8)

plt.fill_between(x_axis, general_log.mean(axis=1) + general_log.std(axis=1),
                 general_log.mean(axis=1) - general_log.std(axis=1),
                 color="black", alpha=0.1)

plt.fill_between(x_axis, invariant_log.mean(axis=1) + invariant_log.std(axis=1),
                 invariant_log.mean(axis=1) - invariant_log.std(axis=1), color="tab:blue",
                 alpha=0.1)

plt.fill_between(x_axis, hybrid_log.mean(axis=1) + hybrid_log.std(axis=1),
                 hybrid_log.mean(axis=1) - hybrid_log.std(axis=1), color="tab:red",
                 alpha=0.1)

plt.plot(x_axis, general_log.mean(axis=1), label="PFN General Classifier", color="black")
plt.plot(x_axis, invariant_log.mean(axis=1), label="Invariant Classifier", color="tab:blue")
plt.plot(x_axis, hybrid_log.mean(axis=1), label="Hybrid Classifier", color="tab:red")

plt.xlabel("Epoch")
plt.ylabel("AUC")

# plt.title("AUC vs. epoch (blur = 10.0%)")
plt.legend(loc="lower right")
plt.savefig('figures/auc vs epoch.pdf', dpi=600)
plt.show()

exit()

# AUC vs. Dataset proportion

n_ensembles = 5

general_db_prop_log = pl.read_csv("classifiers/general/bulk training logs/auc_vs_db_size_*.csv").select(
	["final_auc"]).to_numpy().reshape(n_ensembles, -1).transpose()
invariant_db_prop_log = pl.read_csv("classifiers/invariant/bulk training logs/auc_vs_db_size_*").select(
	["final_auc"]).to_numpy().reshape(n_ensembles, -1).transpose()
hybrid_db_prop_log = pl.read_csv("classifiers/hybrid/bulk training logs/auc_vs_db_size_*").select(
	["final_auc"]).to_numpy().reshape(n_ensembles, -1).transpose()

x_axis = [i / 10 for i in range(1, 11)]

figure = plt.figure()
figure.set_size_inches(12, 8)

plt.fill_between(x_axis, general_db_prop_log.mean(axis=1) + general_db_prop_log.std(axis=1),
                 general_db_prop_log.mean(axis=1) - general_db_prop_log.std(axis=1),
                 color="black", alpha=0.1)

plt.fill_between(x_axis, invariant_db_prop_log.mean(axis=1) + invariant_db_prop_log.std(axis=1),
                 invariant_db_prop_log.mean(axis=1) - invariant_db_prop_log.std(axis=1), color="tab:blue",
                 alpha=0.1)

plt.fill_between(x_axis, hybrid_db_prop_log.mean(axis=1) + hybrid_db_prop_log.std(axis=1),
                 hybrid_db_prop_log.mean(axis=1) - hybrid_db_prop_log.std(axis=1), color="tab:red",
                 alpha=0.1)

plt.plot(x_axis, general_db_prop_log.mean(axis=1), label="PFN General Classifier", color="black")
plt.plot(x_axis, invariant_db_prop_log.mean(axis=1), label="Invariant Classifier", color="tab:blue")
plt.plot(x_axis, hybrid_db_prop_log.mean(axis=1), label="Hybrid Classifier", color="tab:red")

plt.xlabel("Train set size (proportion)")
plt.ylabel("AUC")

# plt.title("AUC vs. Train set size (proportion) (blur = 10.0%)")
plt.legend(loc="lower right")
plt.savefig('figures/mean auc vs train set proportion (x5 dataset).pdf', dpi=600)
plt.show()

# Comparison of mass vs. output for different slices of pT (each slice is shown together)

general_output = pl.read_csv("classifiers/general/analysis.csv").select(["muon_inv_mass", "output", "pT"])
invariant_output = pl.read_csv("classifiers/invariant/analysis.csv").select(["muon_inv_mass", "output", "pT"])
hybrid_output = pl.read_csv("classifiers/hybrid/pT_vs_acc_analysis.csv").select(["muon_inv_mass", "output", "pT"])

figure, axis = plt.subplots(1, 3, sharex=True, sharey=True)

figure.suptitle("Comparison of mass vs. output for different slices of pT (blur = 10.0%)")
figure.supxlabel("Mass")
figure.supylabel("Model output + sigmoid")

output_titles = ["General PFN Classifier", "Invariant Classifier", "Hybrid Classifier"]
outputs = [general_output, invariant_output, hybrid_output]
colors = ["tab:blue", "tab:orange", "tab:green"]
intervals = [0, 30, 40, 300]

plt.rcParams["legend.markerscale"] = 10

for i in range(3):
	for j in range(3):
		axis[i].scatter(
			outputs[i].filter(pl.col("pT").is_between(intervals[j], intervals[j + 1])).to_numpy()[..., 0],
			outputs[i].filter(pl.col("pT").is_between(intervals[j], intervals[j + 1])).to_numpy()[..., 1],
			label=f"{intervals[j]} < pT < {intervals[j + 1]}", color=colors[j], alpha=0.2, s=0.3)
	
	axis[i].set_title(output_titles[i])
	axis[i].legend(loc="upper right")
	axis[i].set_xlim((0.0, 200.0))
	axis[i].set_ylim((0.0, 1.0))

figure.set_size_inches(12, 8)
plt.savefig("figures/mass vs output (pT slices together) (x5 dataset).pdf", dpi=600)
plt.show()

# Comparison of mass vs. output for different slices of pT (each slice is shown separately)

figure, axis = plt.subplots(3, 3, sharex=True, sharey=True)

figure.suptitle("Comparison of mass vs. output for different slices of pT (blur = 10.0%)")
figure.supxlabel("Mass")
figure.supylabel("Model output + sigmoid")

for i in range(3):
	for j in range(3):
		axis[i, j].scatter(
			outputs[j].filter(pl.col("pT").is_between(intervals[i], intervals[i + 1])).to_numpy()[..., 0],
			outputs[j].filter(pl.col("pT").is_between(intervals[i], intervals[i + 1])).to_numpy()[..., 1],
			c=colors[j], s=0.1, marker=".")
		axis[i, j].set_title(f"{output_titles[j]} ({intervals[i]} < pT < {intervals[i + 1]})")

figure.set_size_inches(12, 8)
plt.savefig("figures/mass vs output (pT slices separate) (x5 dataset).pdf", dpi=600)
plt.show()
