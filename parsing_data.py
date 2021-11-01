import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy
import math
import matplotlib.pyplot as plt

reptar_side1_file = 'RePTaR_trials_Aug2021_version1Nov2021.xlsx'
reptar_side1 = pd.read_excel(reptar_side1_file, sheet_name = 'VIDEOS')

print("imported side 1 trials")

reptar_side2_file = 'RePTaRtrials_August2021_8_5_Side2.xlsx'
reptar_side2 = pd.read_excel(reptar_side2_file, sheet_name = 'Trials')

print("imported side 2 trials")

dates_side1 = reptar_side1['Date']
side_nums_side1 = reptar_side1['Side (1/2)']
scan_times_side1 = reptar_side1['AdjTimeSide']
scan_read_bool_side1 = reptar_side1['Read (yes/no)']
scan_distances_side1 = reptar_side1['ApproxDist(in)']

dates_side2 = reptar_side2['Date']
side_nums_side2 = reptar_side2['Side (1/2)']
scan_times_side2 = reptar_side2['AdjTimeSide']
scan_read_bool_side2 = reptar_side2['Read (yes/no)']
scan_distances_side2 = reptar_side2['ApproxDist(in)']

successful_scan_distances_side1 = []
successful_scan_distances_side2 = []

for idx, val in enumerate(scan_read_bool_side1):
    if val == 'yes':
        if not math.isnan(scan_distances_side1[idx]):
            successful_scan_distances_side1.append(scan_distances_side1[idx])

for idx, val in enumerate(scan_read_bool_side2):
    if val == 'yes':
        if not math.isnan(scan_distances_side2[idx]):
            successful_scan_distances_side2.append(scan_distances_side2[idx])

alpha = .05
confidence = 1 - alpha
z_alpha = stats.norm.ppf(confidence)

# For our null hypothesis PDF
delta_0 = 0

n_side1 = len(successful_scan_distances_side1)
n_side2 = len(successful_scan_distances_side2)
mean_side1 = np.mean(successful_scan_distances_side1)
mean_side2 = np.mean(successful_scan_distances_side2)
std_side1 = np.std(successful_scan_distances_side1)
std_side2 = np.std(successful_scan_distances_side2)

stat, p_val = stats.ranksums(successful_scan_distances_side2, successful_scan_distances_side1, alternative='greater')
print("Wilcoxon Rank Sum P value: {}".format(np.round(p_val,4)))

whit_stat, p_norm = stats.mannwhitneyu(successful_scan_distances_side2, successful_scan_distances_side1, use_continuity=False, method="asymptotic", alternative='greater')
print("Mann Whitney U P value: {}".format(np.round(p_norm,4)))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
fig.suptitle("(Fig. 1) Scan Distances for Both Observers")
axes[0].hist(successful_scan_distances_side1)
axes[0].set_xlim([0, 3])
axes[0].set_xticks(np.arange(0, 3.5, 0.5))
axes[0].set_xlabel('Scan Distance (inch)')
axes[0].set_ylabel('Count')
axes[0].title.set_text('Observer 1 Scan Distances');
axes[1].hist(successful_scan_distances_side2)
axes[1].set_xlabel('Scan Distance (inch)')
axes[1].set_ylabel('Count')
axes[1].title.set_text('Observer 2 Scan Distances');
plt.show()



#sauk_years = reptar_side2_data['water_year']
#sauk_peak_flow = reptar_side2_data['peak_va']
