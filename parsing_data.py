import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy
import math
import matplotlib.pyplot as plt
import datetime
import sys
import os
import glob
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod import families

# Import all files
reptar_side1_file = 'RePTaR_trials_Aug2021_version1Nov2021.xlsx'
reptar_side1 = pd.read_excel(reptar_side1_file, sheet_name = 'VIDEOS')
reptar_animal_info = pd.read_excel(reptar_side1_file, sheet_name = 'ANIMALINFO')
reptar_side2_file = 'RePTaRtrials_Aadu.xlsx'
reptar_side2 = pd.read_excel(reptar_side2_file, sheet_name = 'Trials')

# Split data by column (File 1)
dates_side1 = reptar_side1['Date']
cumu_time_side1 = reptar_side1['CumulativeTime']
side_nums_side1 = reptar_side1['Side (1/2)']
scan_times_side1 = reptar_side1['AdjTimeSide']
scan_read_bool_side1 = reptar_side1['Read (yes/no)']
scan_distances_side1 = reptar_side1['ApproxDist(in)']
pits_side1 = reptar_side1['PITTAG']

# Split data by column (File 2)
dates_side2 = reptar_side2['Date']
side_nums_side2 = reptar_side2['Side (1/2)']
scan_times_side2 = reptar_side2['AdjTimeSide']
scan_read_bool_side2 = reptar_side2['Read (yes/no)']
scan_distances_side2 = reptar_side2['ApproxDist(in)']
pits_side2 = reptar_side2['PITTAG']

# This is a new column that I created in Excel
cumu_time_side2 = reptar_side2['CumulativeTime']


# Grab relevant data on ALL snakes
svl_list = reptar_animal_info['SVL']
pit_list = reptar_animal_info['PITTAG']
date_list = reptar_animal_info['DATETEST']
sex_list = reptar_animal_info['SEX']
bulge_list = reptar_animal_info['BULGE']
weight_list = reptar_animal_info['WEIGHT']
days_list = reptar_animal_info['DAYSINLAB']
numscans_list = reptar_animal_info['NUMSCANS']

id_dict = {}
svl_dict = {}
sex_dict = {}
weight_dict = {}
bulge_dict = {}
num_scan_dict = {}

path = 'num_scan_data'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))

# Log the number of scans for each snake and store in a dict with a tuple key
# value of trial date and the specific snake's PIT ID.
for filename in result:
    df = pd.read_csv (filename)
    individual_scans = df["RFID"].value_counts()
    trial_date = df["Date"][0]

    for rfid, scans in individual_scans.items():
        num_scan_dict[(trial_date, rfid)] = scans

male_numscans = []
female_numscans = []

male_weights = []
female_weights = []

total_ids = []
total_svls = []
total_numscans = []
total_weights = []
total_sex = []

# Store each snake's data in relevant dictionary (for each of the features we
# care about. Assign a unique ID for each snake
id = 0
for idx, date in enumerate(date_list):
    date1 = date.strftime("%m/%d")
    svl_dict[(date1, pit_list[idx])] = svl_list[idx]
    weight_dict[(date1, pit_list[idx])] = weight_list[idx]
    sex_dict[(date1, pit_list[idx])] = sex_list[idx]
    bulge_dict[(date1, pit_list[idx])] = bulge_list[idx]
    id_dict[(date1, pit_list[idx])] = id

    # skip snakes with blank values
    try:
        total_numscans.append(num_scan_dict[date1, pit_list[idx]])
        total_svls.append(svl_list[idx])
        total_ids.append(id)
        total_weights.append(weight_list[idx])
    except:
        print("Skipping Snake")

    # Split up male/female snakes into separate sex-dicts
    if sex_list[idx] == 'M':
        total_sex.append(0)
        try:
            male_numscans.append(num_scan_dict[date1, pit_list[idx]])
            male_weights.append(weight_dict[date1, pit_list[idx]])
        except KeyError:
            print("Skipping PIT tag")
    else:
        total_sex.append(1)
        try:
            female_numscans.append(num_scan_dict[date1, pit_list[idx]])
            female_weights.append(weight_dict[date1, pit_list[idx]])
        except KeyError:
            print("Skipping PIT tag")

    id += 1

# Run Shapiro test to assess normality
# stat1, pval = stats.shapiro(svl_list)
# print(pval)

successful_scan_distances_side1 = []
successful_scan_distances_side2 = []
scan_svls = []
scan_sides = []

timeons = []
log_timeons = []
timeon_svls = []
timeon_ids = []

male_scans = []
female_scans = []

male_timeons = []
female_timeons = []

distances_posterior = []
distances_anterior = []

succesful_scan_times = []

categorical_sex = []

glm_loc = []
glm_svl = []
glm_success = []

male_distance_dict = {}
female_distance_dict = {}

unique_svl = []

posterior_distance_dict = {}
anterior_distance_dict = {}

# Iterate through all CSV data and populate
for idx, val in enumerate(scan_read_bool_side1):
    pit = pits_side1[idx]
    if not math.isnan(pit):
        dt = pd.to_datetime(dates_side1[idx])
        date1 = dt.strftime("%m/%d")

        svl = svl_dict[(date1, int(pit))]
        id = id_dict[(date1, int(pit))]
        sex = sex_dict[(date1, int(pit))]
        glm_svl.append(svl)
        if not svl in unique_svl:
            unique_svl.append(svl)

        if not pd.isna(scan_distances_side1[idx]):
            # Add scans to appropriate M/F dict list
            if sex == 'M':
                if id in male_distance_dict and not pd.isna(scan_distances_side1[idx]):
                    male_distance_dict[id].extend([scan_distances_side1[idx]])
                else:
                    male_distance_dict[id] = [scan_distances_side1[idx]]
            else:
                if id in female_distance_dict and not pd.isna(scan_distances_side1[idx]):
                    female_distance_dict[id].extend([scan_distances_side1[idx]])
                else:
                    female_distance_dict[id] = [scan_distances_side1[idx]]

            # We know the posterior trials took place on these dates
            if date1 == '08/28' or date1 == '08/29':
                if id in posterior_distance_dict and not pd.isna(scan_distances_side1[idx]):
                    posterior_distance_dict[id].extend([scan_distances_side1[idx]])
                else:
                    posterior_distance_dict[id] = [scan_distances_side1[idx]]
            else:
                if id in anterior_distance_dict and not pd.isna(scan_distances_side1[idx]):
                    anterior_distance_dict[id].extend([scan_distances_side1[idx]])
                else:
                    anterior_distance_dict[id] = [scan_distances_side1[idx]]

        time = cumu_time_side1[idx]
        if not isinstance(time, float):
            pt = datetime.datetime.strptime(time,'%H:%M:%S')
            total_seconds = pt.second + pt.minute*60 + pt.hour*3600

            # Only add to list if first time
            if total_seconds not in timeons:
                timeons.append(total_seconds)
                log_timeons.append(np.log(total_seconds))
                timeon_svls.append(svl)
                timeon_ids.append(id)
        else:
            total_seconds = 0

        # Add time on trap to M/F lists
        if sex_dict[(date1, int(pit))] == 'M':
            male_scans.append(scan_distances_side1[idx])
            categorical_sex.append(0)
            if not total_seconds is 0 and total_seconds not in male_timeons:
                male_timeons.append(total_seconds)
        else:
            female_scans.append(scan_distances_side1[idx])
            categorical_sex.append(1)
            if not total_seconds is 0 and total_seconds not in female_timeons:
                female_timeons.append(total_seconds)

        if date1 == '08/28' or date1 == '08/29':
            glm_loc.append(1)
        else:
            glm_loc.append(0)

        if val == 'yes':
            glm_success.append(1)
        else:
            glm_success.append(0)

# Repeat for 2nd file
for idx, val in enumerate(scan_read_bool_side2):
    pit = pits_side2[idx]
    if not math.isnan(pit):

        dt = pd.to_datetime(dates_side2[idx])
        date1 = dt.strftime("%m/%d")

        svl = svl_dict[(date1, int(pit))]
        id = id_dict[(date1, int(pit))]
        sex = sex_dict[(date1, int(pit))]
        glm_svl.append(svl)
        if not svl in unique_svl:
            unique_svl.append(svl)

        if not pd.isna(scan_distances_side2[idx]):
            if sex == 'M':
                if id in male_distance_dict:
                    male_distance_dict[id].extend([scan_distances_side2[idx]])
                else:
                    male_distance_dict[id] = [scan_distances_side2[idx]]
            else:
                if id in female_distance_dict:
                    female_distance_dict[id].extend([scan_distances_side2[idx]])
                else:
                    female_distance_dict[id] = [scan_distances_side2[idx]]

            if date1 == '08/28' or date1 == '08/29':
                if id in posterior_distance_dict and not pd.isna(scan_distances_side2[idx]):
                    posterior_distance_dict[id].extend([scan_distances_side2[idx]])
                else:
                    posterior_distance_dict[id] = [scan_distances_side2[idx]]
            else:
                if id in anterior_distance_dict and not pd.isna(scan_distances_side2[idx]):
                    anterior_distance_dict[id].extend([scan_distances_side2[idx]])
                else:
                    anterior_distance_dict[id] = [scan_distances_side2[idx]]

        time = cumu_time_side2[idx]
        if not isinstance(time, float):
            pt = datetime.datetime.strptime(time,'%H:%M:%S')
            total_seconds = pt.second + pt.minute*60 + pt.hour*3600
            if total_seconds not in timeons:
                timeons.append(total_seconds)
                log_timeons.append(np.log(total_seconds))
                timeon_svls.append(svl)
                timeon_ids.append(id)
        else:
            total_seconds = 0

        if sex_dict[(date1, int(pit))] == 'M':
            male_scans.append(scan_distances_side2[idx])
            categorical_sex.append(0)
            if not total_seconds is 0 and total_seconds not in male_timeons:
                male_timeons.append(total_seconds)
        else:
            female_scans.append(scan_distances_side2[idx])
            categorical_sex.append(1)
            if not total_seconds is 0 and total_seconds not in female_timeons:
                female_timeons.append(total_seconds)

        if date1 == '08/28' or date1 == '08/29':
            glm_loc.append(1)
        else:
            glm_loc.append(0)

        if val == 'yes':
            glm_success.append(1)
        else:
            glm_success.append(0)

# stat1, pval = stats.shapiro(unique_svl)
# print("Total Scans Shapiro Test p-val: ", pval)
# sys.exit(0)
#
# # 32 individuals
# print(len(male_distance_dict))
# print(len(female_distance_dict))
#
posterior_distance_list = []
for key,value in posterior_distance_dict.items() :
    posterior_distance_list.append(np.average(value))

anterior_distance_list = []
for key,value in anterior_distance_dict.items() :
    anterior_distance_list.append(np.average(value))

print(posterior_distance_list)
print(anterior_distance_list)

stat, p_val = stats.ranksums(posterior_distance_list, \
anterior_distance_list, alternative='less')

print("\nAverage Posterior Scan Distance: ", np.average(posterior_distance_list))
print("Average Anterior Scan Distance: ", np.average(anterior_distance_list))
print("Wilcoxon Rank Sum p-value: ", p_val)
print("\n")

sys.exit(0)


# df = pd.DataFrame(list(zip(glm_success, glm_svl, glm_loc)), columns =['Success', 'SVL', 'Loc'])
# md = smf.glm("Success ~ C(Loc) + SVL + C(Loc)*SVL", df, family=families.Binomial())
# mdf = md.fit()
# #print(mdf.random_effects)
# print(mdf.summary())
# sys.exit(0)

# for idx, val in enumerate(scan_read_bool_side1):
    # if val == 'yes':
    #     succesful_scan_times.append(scan_times_side1[idx])
    #     if not math.isnan(scan_distances_side1[idx]):
    #         successful_scan_distances_side1.append(scan_distances_side1[idx])
    #         scan_sides.append(side_nums_side1[idx])
    #         pit = pits_side1[idx]
    #         if not math.isnan(pit):
    #             dt = pd.to_datetime(dates_side1[idx])
    #             date1 = dt.strftime("%m/%d")
    #             time1 = scan_times_side1[idx].strftime("%H:%M:S")
    #             succesful_scan_times.append(time1)
    #             svl = svl_dict[(date1, int(pit))]
    #             unique_id = id_dict[(date1, int(pit))]
    #             scan_svls.append(svl)
    #
    #             if date1 == '08/28' or date1 == '08/29':
    #                 distances_posterior.append(scan_distances_side1[idx])
    #             else:
    #                 distances_anterior.append(scan_distances_side1[idx])
    #
    #             time = cumu_time_side1[idx]
    #             if not isinstance(time, float):
    #                 pt = datetime.datetime.strptime(time,'%H:%M:%S')
    #                 total_seconds = pt.second + pt.minute*60 + pt.hour*3600
    #                 if total_seconds not in timeons:
    #                     timeons.append(total_seconds)
    #                     log_timeons.append(np.log(total_seconds))
    #                     timeon_svls.append(svl)
    #                     timeon_ids.append(unique_id)
    #             else:
    #                 total_seconds = 0
    #
    #             if sex_dict[(date1, int(pit))] == 'M':
    #                 male_scans.append(scan_distances_side1[idx])
    #                 categorical_sex.append(0)
    #                 if not total_seconds is 0 and total_seconds not in male_timeons:
    #                     male_timeons.append(total_seconds)
    #             else:
    #                 female_scans.append(scan_distances_side1[idx])
    #                 categorical_sex.append(1)
    #                 if not total_seconds is 0 and total_seconds not in female_timeons:
    #                     female_timeons.append(total_seconds)

# for idx, val in enumerate(scan_read_bool_side2):
    # if val == 'yes':
    #     if not math.isnan(scan_distances_side2[idx]):
    #         successful_scan_distances_side2.append(scan_distances_side2[idx])
    #         scan_sides.append(side_nums_side2[idx])
    #         pit = pits_side2[idx]
    #         if not math.isnan(pit):
    #             dt = pd.to_datetime(dates_side2[idx])
    #             date2 = dt.strftime("%m/%d")
    #             time1 = scan_times_side2[idx].strftime("%H")
    #             succesful_scan_times.append(time1)
    #             svl = svl_dict[(date2, int(pit))]
    #             unique_id = id_dict[(date2, int(pit))]
    #             scan_svls.append(svl)
    #
    #             if date2 == '08/28' or date2 == '08/29':
    #                 distances_posterior.append(scan_distances_side2[idx])
    #             else:
    #                 distances_anterior.append(scan_distances_side2[idx])
    #
    #             time = cumu_time_side2[idx]
    #             if not isinstance(time, float):
    #                 pt = datetime.datetime.strptime(time,'%H:%M:%S')
    #                 total_seconds = pt.second + pt.minute*60 + pt.hour*3600
    #                 if total_seconds not in timeons:
    #                     timeons.append(total_seconds)
    #                     log_timeons.append(np.log(total_seconds))
    #                     timeon_svls.append(svl)
    #                     timeon_ids.append(unique_id)
    #             else:
    #                 total_seconds = 0
    #
    #             if sex_dict[(date2, int(pit))] == 'M':
    #                 male_scans.append(scan_distances_side2[idx])
    #                 categorical_sex.append(0)
    #                 if not total_seconds is 0 and total_seconds not in male_timeons:
    #                     male_timeons.append(total_seconds)
    #             else:
    #                 female_scans.append(scan_distances_side2[idx])
    #                 categorical_sex.append(1)
    #                 if not total_seconds is 0 and total_seconds not in female_timeons:
    #                     female_timeons.append(total_seconds)

df = pd.DataFrame(list(zip(timeon_ids, categorical_sex, timeon_svls, log_timeons)), columns =['ID', 'Sex', 'SVL', 'TimeOn'])
md = smf.mixedlm("TimeOn ~ SVL + C(Sex)", df, groups=df["ID"])
mdf = md.fit()
#print(mdf.random_effects)
print(mdf.summary())
sys.exit(0)

# df = pd.DataFrame(list(zip(timeon_ids, categorical_sex, timeon_svls, log_timeons)), columns =['ID', 'Sex', 'SVL', 'TimeOn'])
# md = smf.ols("TimeOn ~ SVL", df)
# mdf = md.fit()
# print(mdf.summary())
# sys.exit(0)

# df = pd.DataFrame(list(zip(timeon_ids, categorical_sex, timeon_svls, log_timeons)), columns =['ID', 'Sex', 'SVL', 'TimeOn'])
# md = smf.mixedlm("TimeOn ~ SVL + C(Sex)", df, groups=df["ID"])
# mdf = md.fit()
# print(mdf.summary())
# sys.exit(0)

log_numscans = np.log(total_numscans)
slope, intercept, rvalue, pvalue, stderr = stats.linregress(total_svls, total_numscans);

print('B0 : {}'.format(np.round(intercept,4)))
print('B1 : {}'.format(np.round(slope,4)))
print('R^2 : {}'.format(np.round(rvalue**2,3)))
print('R : {}'.format(np.round(rvalue,3)))
print('pvalue : {}'.format(np.round(pvalue,5)))

sys.exit(0)

# Create points for the regression line
# x = np.linspace(np.min(total_svls), np.max(total_svls), 2) # make two x coordinates from min and max values of SLI_max
# y = slope * x + intercept # y coordinates using the slope and intercept from our linear regression to draw a regression line
#
# predicted_numscans = []
# for val in total_svls:
#     y_hat = intercept + (slope*val)
#     predicted_numscans.append(y_hat)
#
# linear_residuals = []
# for idx, y_real in enumerate(log_numscans):
#     linear_residuals.append(y_real - predicted_numscans[idx])
#
# fig = plt.figure(figsize=(5,5))
# axis = fig.add_subplot(111)
# axis.plot(x, y, '-r', label="Linear Regression")
# plt.scatter(total_svls, log_numscans, c='k')
# plt.xlabel('Scanned Snout-to-Vent Length (mm)');
# plt.ylabel('Log(Number of Scans)');
# plt.title('Linear Regression of Log(Number of Scans) vs Snout-to-Vent Length (SVL)')
# plt.legend();
# plt.show()
#
# fig = plt.figure(figsize=(5,5))
# axis = fig.add_subplot(111)
# plt.scatter(predicted_numscans, linear_residuals, c='k')
# plt.xlabel('Fitted Number of Scans');
# plt.ylabel('Linear Residual');
# plt.title('Residual Analysis of Linear Regression')
# plt.legend();
# plt.show()
#
df = pd.DataFrame(list(zip(total_ids, total_sex, total_svls, log_numscans)), columns =['ID', 'Sex', 'SVL', 'Log_NumScans'])
md = smf.ols("Log_NumScans ~ SVL + C(Sex)", df)
mdf = md.fit()
print(mdf.summary())
sys.exit(0)

# df = pd.DataFrame(list(zip(total_ids, total_sex, total_weights, total_svls, log_numscans)), columns =['ID', 'Sex', 'Weight', 'SVL', 'Log_NumScans'])
# md = smf.ols("Log_NumScans ~ SVL + Weight", df)
# mdf = md.fit()
# #print(mdf.random_effects)
# print(mdf.summary())
# sys.exit(0)

# print("\n\
# ----------------------------------------------------------------------------\n\
# - Calculate whether Male/Female Scan Numbers are Statistically Different   -\n\
# ----------------------------------------------------------------------------\n\
# \n")
# #
# print("Mean Male Num Scans: ", np.mean(male_numscans))
# print("Mean Female Num Scans: ", np.mean(female_numscans))
#
# stat1, pval = stats.shapiro(total_numscans)
# print("Total Scans Shapiro Test p-val: ", pval)
# stat1, pval = stats.shapiro(male_numscans)
# print("Male Scans Shapiro Test p-val: ", pval)
# stat1, pval = stats.shapiro(female_numscans)
# print("Female Scans Shapiro Test p-val: ", pval)
#
# stat, p_val = stats.ranksums(male_numscans, \
# female_numscans, alternative='less')
#
# print("Wilcoxon Rank Sum p-value: ", p_val)


# print("\n\
# ----------------------------------------------------------------------------\n\
# - Calculate whether Male/Female Scan Distances are Statistically Different -\n\
# ----------------------------------------------------------------------------\n\
# ")
# total_scans = male_scans + female_scans
#
# print("\nMean Male Weights: ", np.mean(male_weights))
# print("Len Male Weights: ", len(male_weights))
# print("Mean Female Weights: ", np.mean(female_weights))
# print("Len Female Weights: ", len(female_weights))
#
# stat1, pval = stats.shapiro(total_weights)
# print("Total Weights Shapiro Test p-val: ", pval)
# stat1, pval = stats.shapiro(male_weights)
# print("Male Weights Shapiro Test p-val: ", pval)
# stat1, pval = stats.shapiro(female_weights)
# print("Female Weights Shapiro Test p-val: ", pval)
#
# # stat, p_val = stats.ttest_ind(male_scans, \
# # female_scans)
# #
# # print("T-test P value: ", p_val)
# # print("\n")
#
# stat, p_val = stats.ranksums(male_weights, \
# female_weights, alternative='less')
#
# print("Wilcoxon Rank Sum P value (Weight): ", p_val)
# print("\n")
#
# print("\nMean Male Scan Distances: ", np.mean(male_scans))
# print("Mean Female Scan Distances: ", np.mean(female_scans))
# stat1, pval = stats.shapiro(total_scans)
# print("Total Scans Shapiro Test p-val: ", pval)
# stat1, pval = stats.shapiro(male_scans)
# print("Male Scans Shapiro Test p-val: ", pval)
# stat1, pval = stats.shapiro(female_scans)
# print("Female Scans Shapiro Test p-val: ", pval)
#
# stat, p_val = stats.ranksums(male_scans, \
# female_scans, alternative='greater')
#
# print("Wilcoxon Rank Sum P value (Scan Distances): ", p_val)
# print("\n")
#
# print("\n\
# --------------------------------------------------------------------------\n\
# - Calculate whether Male/Female Time On Trap are Statistically Different -\n\
# --------------------------------------------------------------------------\n\
# ")
#
# stat, p_val = stats.ranksums(male_timeons, \
# female_timeons, alternative='greater')
#
# labels = ['Male Time on Trap', 'Female Time on Trap']
# bins_m = int(np.sqrt(len(male_timeons)))
# bins_f = int(np.sqrt(len(female_timeons)))
#
# print(len(male_timeons))
# print(len(female_timeons))
#
# # stats f_oneway functions takes the groups as input and returns an F and P-value\n",
# fvalue, pvalue = stats.f_oneway(male_timeons, female_timeons)
#
# # print the results\n
# print("F-statistic = {}".format( np.round(fvalue,2)))
# print("p = {}".format( pvalue ))
#
# print("Mean Male Time on Trap (s): ", np.mean(male_timeons))
# print("Mean Female Time on Trap (s): ", np.mean(female_timeons))
#
# print("Wilcoxon Rank Sum P value: ", p_val)
#
# whit_stat, p_norm = stats.mannwhitneyu(male_timeons,\
#  female_timeons, use_continuity=True, method="asymptotic",\
#   alternative='greater')
#
# print("Mann Whitney U P value: {}".format(np.round(p_norm,4)))


# print("\n\
# --------------------------------------------------------------------------\n\
# - Plot Regression of SVL vs Num Scans -\n\
# --------------------------------------------------------------------------\n\
# ")
# pruned_svls = []
# pruned_numscans = []
#
# for idx,val in enumerate(numscans_list):
#     if not math.isnan(val):
#         pruned_numscans.append(val)
#         pruned_svls.append(svl_list[idx])
#
# slope, intercept, rvalue, pvalue, stderr = stats.linregress(pruned_svls, pruned_numscans);
#
# print('B0 : {}'.format(np.round(intercept,4)))
# print('B1 : {}'.format(np.round(slope,4)))
# print('R^2 : {}'.format(np.round(rvalue**2,3)))
# print('R : {}'.format(np.round(rvalue,3)))
# print('pvalue : {}'.format(np.round(pvalue,3)))
#
# # logreg = LogisticRegression()
# # logreg.fit(pruned_svls, pruned_numscans)
# # yhat = model.predict(pruned_svls)
# # SS_Residual = sum((pruned_numscans-yhat)**2)
# # SS_Total = sum((pruned_numscans-np.mean(pruned_numscans))**2)
# # r_squared = 1 - (float(SS_Residual))/SS_Total
# # adjusted_r_squared = 1 - (1-r_squared)*(len(pruned_numscans)-1)/(len(pruned_numscans)-pruned_svls.shape[1]-1)
# # print (r_squared, adjusted_r_squared)
#
# # Create points for the regression line
# x = np.linspace(np.min(pruned_svls), np.max(pruned_svls), 2) # make two x coordinates from min and max values of SLI_max
# y = slope * x + intercept # y coordinates using the slope and intercept from our linear regression to draw a regression line
#
# fig = plt.figure(figsize=(5,5))
# axis = fig.add_subplot(111)
# axis.plot(x, y, '-r', label="Linear Regression")
# plt.scatter(timeon_svls, timeons, c='k')
# plt.xlabel('Scanned Snout-to-Vent Length (mm)');
# plt.ylabel('Time Spent on Trap (s)');
# plt.title(' (Fig. 2) Linear Regression of SVL vs Time Spent on Trap')
# plt.legend();
# plt.show()


# print("\n\
# --------------------------------------------------------------------------\n\
# - Plot Regression of SVL vs Time on Trap -\n\
# --------------------------------------------------------------------------\n\
# ")



# log_timeons = np.log(timeons)
# slope, intercept, rvalue, pvalue, stderr = stats.linregress(timeon_svls, timeons);
#
# print('B0 : {}'.format(np.round(intercept,4)))
# print('B1 : {}'.format(np.round(slope,4)))
# print('R^2 : {}'.format(np.round(rvalue**2,3)))
# print('R : {}'.format(np.round(rvalue,3)))
# print('pvalue : ', pvalue)
#
# # Create points for the regression line
# x = np.linspace(np.min(timeon_svls), np.max(timeon_svls), 2) # make two x coordinates from min and max values of SLI_max
# y = slope * x + intercept # y coordinates using the slope and intercept from our linear regression to draw a regression line
#
# predicted_timeons = []
#
# for val in timeon_svls:
#     y_hat = intercept + (slope*val)
#     predicted_timeons.append(y_hat)
#
# linear_residuals = []
# for idx, y_real in enumerate(timeons):
#     linear_residuals.append(y_real - predicted_timeons[idx])
#
# fig = plt.figure(figsize=(5,5))
# axis = fig.add_subplot(111)
# axis.plot(x, y, '-r', label="Linear Regression")
# plt.scatter(timeon_svls, timeons, c='k')
# plt.xlabel('Scanned Snout-to-Vent Length (mm)');
# plt.ylabel('Time Spent on Trap');
# plt.title('Linear Regression of Time Spent on Trap vs SVL')
# plt.legend();
# plt.show()
#
# fig = plt.figure(figsize=(5,5))
# axis = fig.add_subplot(111)
# plt.scatter(predicted_timeons, linear_residuals, c='k')
# plt.xlabel('Fitted Time on Trap');
# plt.ylabel('Linear Residual');
# plt.title('Residual Analysis of Linear Regression')
# plt.legend();
# plt.show()
