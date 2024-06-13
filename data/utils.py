import numpy as np
import polars as pl


def convert_from_lhe(file_path: str, limit: int | None = None) -> list[dict]:
	"""
	Converts an LHE file to a list containing the information from each event.

	Args:
		file_path: Path to the LHE file.
		limit: Maximum number of events to process.

	Returns:
		list[dict]: List of dictionary elements containing information about each event.
	"""
	events = []
	
	with open(file_path, "r") as data:
		data = list(map(str.strip, data.readlines()))
		event_start_indices = [i for i, token in enumerate(data) if token == "<event>"]
		
		for start_index in event_start_indices:
			if limit is not None and len(events) > limit:
				break
			
			num_of_particles = int(data[start_index + 1][0])
			
			final_state_muons = []
			final_state_muon_count = 0
			cols = [
				"id",
				"mother1",
				"mother2",
				"color1",
				"color2",
				"px",
				"py",
				"pz",
				"energy",
				"mass",
				"lifetime",
				"spin",
			]
			for particle in data[start_index + 2: start_index + 2 + num_of_particles]:
				particle = particle.split(" ")
				particle = [i for i in particle if i != ""]
				particle = list(map(float, particle))
				if abs(particle[0]) == 13 and particle[1] == 1:
					particle.pop(1)
					particle = list(
						zip(
							map(lambda col: col + f"_{final_state_muon_count}", cols),
							particle,
						)
					)
					final_state_muons.extend(particle)
					final_state_muon_count += 1
			
			event = {k: v for k, v in final_state_muons}
			
			muon_inv_mass = np.sqrt((event["energy_0"] + event["energy_1"]) ** 2
			                        - ((event["px_0"] + event["px_1"]) ** 2
			                           + (event["py_0"] + event["py_1"]) ** 2
			                           + (event["pz_0"] + event["pz_1"]) ** 2))
			
			event["n"] = num_of_particles
			event["muon_inv_mass"] = muon_inv_mass
			events.append(event)
	
	return events


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	background_data = convert_from_lhe(
		"amm_forsym_test_8263464-5_unweighted_events.lhe"
	)
	csv_background_data = pl.DataFrame(background_data)
	csv_background_data.write_csv("background.csv")
	
	print("Saved background.csv")
	
	signal_data = convert_from_lhe("zmm_forsym_test_8263460-1_unweighted_events.lhe")
	csv_signal_data = pl.DataFrame(signal_data)
	csv_signal_data.write_csv("signal.csv")
	
	print("Saved signal.csv")
	
	# amalgam_inv_mass = pl.concat((csv_background_data, csv_signal_data)).get_column("muon_inv_mass").to_numpy()
	plt.hist([csv_background_data.get_column("muon_inv_mass").to_numpy(),
	          csv_signal_data.get_column("muon_inv_mass").to_numpy()], 100)
	plt.show()
