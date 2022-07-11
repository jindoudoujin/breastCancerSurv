from django.http import HttpResponse
import tensorflow as tf
import utilities as Utils
import pandas as pd
import numpy as np
import breast_utilities as br_utils
import json
import matplotlib.pyplot as plt


def hello(request):
    jobj = json.loads(request.body)
    df = Utils.read_from_file("demo/breast_cancer/breast.csv")
    df = Utils.filter_col_data(df, ["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                    "ICD-O-3 Hist/behav",
                                    "Breast - Adjusted AJCC 6th T (1988-2015)",
                                    "Breast - Adjusted AJCC 6th N (1988-2015)",
                                    "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                    "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                    "Laterality", "Breast Subtype (2010+)",
                                    "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                    "Chemotherapy recode (yes, no/unk)",
                                    "End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"])
    # according to https://seer.cancer.gov/icd-o-3/sitetype.icdo3.20220429.pdf
    duct_carcinoma_array = ['8500/3: Infiltrating duct carcinoma, NOS', '8501/3: Comedocarcinoma, NOS',
                            '8502/3: Secretory carcinoma of breast',
                            '8503/3: Intraductal papillary adenocarcinoma with invasion',
                            '8504/3: Intracystic carcinoma, NOS', '8507/3: Ductal carcinoma, micropapillary']
    # according to https://seer.cancer.gov/icd-o-3/sitetype.icdo3.20220429.pdf
    lobular_and_other_ductal_array = ['8520/3: Lobular carcinoma, NOS', '8521/3: Infiltrating ductular carcinoma',
                                      '8522/3: Infiltrating duct and lobular carcinoma',
                                      '8523/3: Infiltrating duct mixed with other types of carcinoma',
                                      '8524/3: Infiltrating lobular mixed with other types of carcinoma',
                                      '8525/3: Polymorphous low grade adenocarcinoma']
    duct_lobular_array = duct_carcinoma_array + lobular_and_other_ductal_array

    # filter the ICD-O-3 Hist/behav whose type is DUCT CARCINOM and LOBULAR AND OTHER DUCTAL CA
    df = Utils.select_data_from_values(df, "ICD-O-3 Hist/behav", duct_lobular_array)

    # map "RX Summ--Surg Prim Site (1998+)" according to map_breast_surg_type
    df = Utils.map_one_col_data(df, "RX Summ--Surg Prim Site (1998+)", br_utils.map_breast_surg_type)

    df = pd.get_dummies(df, prefix=["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                    "ICD-O-3 Hist/behav",
                                    "Breast - Adjusted AJCC 6th T (1988-2015)",
                                    "Breast - Adjusted AJCC 6th N (1988-2015)",
                                    "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                    "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                    "Laterality", "Breast Subtype (2010+)",
                                    "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                    "Chemotherapy recode (yes, no/unk)"],
                        columns=["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                 "ICD-O-3 Hist/behav",
                                 "Breast - Adjusted AJCC 6th T (1988-2015)",
                                 "Breast - Adjusted AJCC 6th N (1988-2015)",
                                 "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                 "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                 "Laterality", "Breast Subtype (2010+)",
                                 "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                 "Chemotherapy recode (yes, no/unk)"])
    Y_train = Utils.filter_col_data(df, ["End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"])
    X_train = df.drop(["End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"], axis=1)
    Y_train = Utils.map_one_col_data(Y_train, "End Calc Vital Status (Adjusted)", br_utils.map_event_code)
    '''
    The S(t) is derived from h(t|X), 
    based on https://www.andrew.cmu.edu/user/georgech/Introduction%20to%20Survival%20Analysis%20Math.pdf
    Just some integration and sigma opeartions
    reference https://github.com/havakv/pycox
    '''
    model = tf.keras.models.load_model('demo/breast_cancer/_temp.h5', compile=False)
    max_duration = np.inf
    base_haz = Y_train.assign(expg=np.exp(model.predict(X_train))).groupby("Number of Intervals (Calculated)").agg(
        {'expg': 'sum', "End Calc Vital Status (Adjusted)": 'sum'}).sort_index(ascending=False).assign(
        expg=lambda x: x['expg'].cumsum()).pipe(lambda x: x["End Calc Vital Status (Adjusted)"] / x['expg']).fillna(
        0.).iloc[::-1].loc[lambda x: x.index <= max_duration].rename('baseline_hazards')
    base_cum_haz = (base_haz
                    .cumsum()
                    .rename('baseline_cumulative_hazards'))
    base_cum_haz = base_cum_haz.loc[lambda x: x.index <= max_duration]
    # entity
    patient = pd.DataFrame(
        {"Age recode with <1 year olds_15-19 years": 0, "Age recode with <1 year olds_20-24 years": 0,
         "Age recode with <1 year olds_25-29 years": 0, "Age recode with <1 year olds_30-34 years": 0,
         "Age recode with <1 year olds_35-39 years": 0, "Age recode with <1 year olds_40-44 years": 0,
         "Age recode with <1 year olds_45-49 years": 0, "Marital status at diagnosis_Divorced": 0,
         "Marital status at diagnosis_Married (including common law)": 0,
         "Marital status at diagnosis_Separated": 0,
         "Marital status at diagnosis_Single (never married)": 0, "Marital status at diagnosis_Unknown": 0,
         "Marital status at diagnosis_Unmarried or Domestic Partner": 0,
         "Marital status at diagnosis_Widowed": 0,
         "Grade (thru 2017)_Moderately differentiated; Grade II": 0,
         "Grade (thru 2017)_Poorly differentiated; Grade III": 0,
         "Grade (thru 2017)_Undifferentiated; anaplastic; Grade IV": 0,
         "Grade (thru 2017)_Well differentiated; Grade I": 0,
         "ICD-O-3 Hist/behav_8500/3: Infiltrating duct carcinoma, NOS": 0,
         "ICD-O-3 Hist/behav_8501/3: Comedocarcinoma, NOS": 0,
         "ICD-O-3 Hist/behav_8502/3: Secretory carcinoma of breast": 0,
         "ICD-O-3 Hist/behav_8503/3: Intraductal papillary adenocarcinoma with invasion": 0,
         "ICD-O-3 Hist/behav_8504/3: Intracystic carcinoma, NOS": 0,
         "ICD-O-3 Hist/behav_8507/3: Ductal carcinoma, micropapillary": 0,
         "ICD-O-3 Hist/behav_8520/3: Lobular carcinoma, NOS": 0,
         "ICD-O-3 Hist/behav_8521/3: Infiltrating ductular carcinoma": 0,
         "ICD-O-3 Hist/behav_8522/3: Infiltrating duct and lobular carcinoma": 0,
         "ICD-O-3 Hist/behav_8523/3: Infiltrating duct mixed with other types of carcinoma": 0,
         "ICD-O-3 Hist/behav_8524/3: Infiltrating lobular mixed with other types of carcinoma": 0,
         "ICD-O-3 Hist/behav_8525/3: Polymorphous low grade adenocarcinoma": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T0": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T1a": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T1b": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T1c": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T1mic": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T2": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T3": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T4a": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T4b": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T4c": 0,
         "Breast - Adjusted AJCC 6th T (1988-2015)_T4d": 0,
         "Breast - Adjusted AJCC 6th N (1988-2015)_N0": 0,
         "Breast - Adjusted AJCC 6th N (1988-2015)_N1": 0,
         "Breast - Adjusted AJCC 6th N (1988-2015)_N2": 0,
         "Breast - Adjusted AJCC 6th N (1988-2015)_N3": 0,
         "Breast - Adjusted AJCC 6th M (1988-2015)_M0": 0, "CS Tumor Size/Ext Eval (2004-2015)_0": 0,
         "CS Tumor Size/Ext Eval (2004-2015)_1": 0, "CS Tumor Size/Ext Eval (2004-2015)_3": 0,
         "CS Tumor Size/Ext Eval (2004-2015)_5": 0, "CS Tumor Size/Ext Eval (2004-2015)_6": 0,
         "CS Reg Node Eval (2004-2015)_0": 0, "CS Reg Node Eval (2004-2015)_1": 0,
         "CS Reg Node Eval (2004-2015)_2": 0, "CS Reg Node Eval (2004-2015)_3": 0,
         "CS Reg Node Eval (2004-2015)_5": 0, "CS Reg Node Eval (2004-2015)_6": 0,
         "CS Mets Eval (2004-2015)_0": 0, "CS Mets Eval (2004-2015)_1": 0, "CS Mets Eval (2004-2015)_3": 0,
         "CS Mets Eval (2004-2015)_5": 0, "CS Mets Eval (2004-2015)_6": 0,
         "Laterality_Bilateral, single primary": 0, "Laterality_Left - origin of primary": 0,
         "Laterality_Only one side - side unspecified": 0,
         "Laterality_Paired site, but no information concerning laterality": 0,
         "Laterality_Right - origin of primary": 0, "Breast Subtype (2010+)_HR+/HER2+ (Luminal B)": 0,
         "Breast Subtype (2010+)_HR+/HER2- (Luminal A)": 0,
         "Breast Subtype (2010+)_HR-/HER2+ (HER2 enriched)": 0,
         "Breast Subtype (2010+)_HR-/HER2- (Triple Negative)": 0,
         "RX Summ--Surg Prim Site (1998+)_Bilateral mastectomy": 0,
         "RX Summ--Surg Prim Site (1998+)_Extended radical mastectomy": 0,
         "RX Summ--Surg Prim Site (1998+)_Local tumor destruction": 0,
         "RX Summ--Surg Prim Site (1998+)_Mastectomy": 0,
         "RX Summ--Surg Prim Site (1998+)_Modified radical mastectomy": 0,
         "RX Summ--Surg Prim Site (1998+)_None": 0,
         "RX Summ--Surg Prim Site (1998+)_Partial mastectomy": 0,
         "RX Summ--Surg Prim Site (1998+)_Radical mastectomy": 0,
         "RX Summ--Surg Prim Site (1998+)_Subcutaneous mastectomy": 0,
         "RX Summ--Surg Prim Site (1998+)_Total (simple) mastectomy": 0,
         "Radiation recode_Beam radiation": 0,
         "Radiation recode_Combination of beam with implants or isotopes": 0,
         "Radiation recode_None/Unknown": 0,
         "Radiation recode_Radiation, NOS  method or source not specified": 0,
         "Radiation recode_Radioactive implants (includes brachytherapy) (1988+)": 0,
         "Radiation recode_Radioisotopes (1988+)": 0,
         "Radiation recode_Recommended, unknown if administered": 0, "Radiation recode_Refused (1988+)": 0,
         "Chemotherapy recode (yes, no/unk)_No/Unknown": 0, "Chemotherapy recode (yes, no/unk)_Yes": 0}, index=[0])
    age = int(jobj["age"])
    if age == 1:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_15-19 years")] = 1
    elif age == 2:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_20-24 years")] = 1
    elif age == 3:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_25-29 years")] = 1
    elif age == 4:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_30-34 years")] = 1
    elif age == 5:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_35-39 years")] = 1
    elif age == 6:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_40-44 years")] = 1
    elif age == 7:
        patient.iloc[0, patient.columns.get_loc("Age recode with <1 year olds_45-49 years")] = 1
    marital_status = int(jobj["marital"])
    if marital_status == 1:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Divorced")] = 1
    elif marital_status == 2:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Married (including common law)")] = 1
    elif marital_status == 3:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Separated")] = 1
    elif marital_status == 4:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Single (never married)")] = 1
    elif marital_status == 5:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Unknown")] = 1
    elif marital_status == 6:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Unmarried or Domestic Partner")] = 1
    elif marital_status == 7:
        patient.iloc[0, patient.columns.get_loc("Marital status at diagnosis_Widowed")] = 1
    grade = int(jobj["grade"])
    if grade == 1:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Well differentiated; Grade I")] = 1
    elif grade == 2:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Moderately differentiated; Grade II")] = 1
    elif grade == 3:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Poorly differentiated; Grade III")] = 1
    elif grade == 4:
        patient.iloc[0, patient.columns.get_loc("Grade (thru 2017)_Undifferentiated; anaplastic; Grade IV")] = 1
    histologic_type = int(jobj["histologic"])
    if histologic_type == 1:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8500/3: Infiltrating duct carcinoma, NOS")] = 1
    elif histologic_type == 2:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8501/3: Comedocarcinoma, NOS")] = 1
    elif histologic_type == 3:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8502/3: Secretory carcinoma of breast")] = 1
    elif histologic_type == 4:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8503/3: Intraductal papillary adenocarcinoma with invasion")] = 1
    elif histologic_type == 5:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8504/3: Intracystic carcinoma, NOS")] = 1
    elif histologic_type == 6:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8507/3: Ductal carcinoma, micropapillary")] = 1
    elif histologic_type == 7:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8520/3: Lobular carcinoma, NOS")] = 1
    elif histologic_type == 8:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8521/3: Infiltrating ductular carcinoma")] = 1
    elif histologic_type == 9:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8522/3: Infiltrating duct and lobular carcinoma")] = 1
    elif histologic_type == 10:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8523/3: Infiltrating duct mixed with other types of carcinoma")] = 1
    elif histologic_type == 11:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8524/3: Infiltrating lobular mixed with other types of carcinoma")] = 1
    elif histologic_type == 12:
        patient.iloc[0, patient.columns.get_loc("ICD-O-3 Hist/behav_8525/3: Polymorphous low grade adenocarcinoma")] = 1
    t = int(jobj["T"])
    if t == 1:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T0")] = 1
    elif t == 2:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T1a")] = 1
    elif t == 3:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T1b")] = 1
    elif t == 4:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T1c")] = 1
    elif t == 5:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T1mic")] = 1
    elif t == 6:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T2")] = 1
    elif t == 7:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T3")] = 1
    elif t == 8:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T4a")] = 1
    elif t == 9:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T4b")] = 1
    elif t == 10:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T4c")] = 1
    elif t == 11:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th T (1988-2015)_T4d")] = 1
    n = int(jobj["N"])
    if n == 1:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th N (1988-2015)_N0")] = 1
    elif n == 2:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th N (1988-2015)_N1")] = 1
    elif n == 3:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th N (1988-2015)_N2")] = 1
    elif n == 4:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th N (1988-2015)_N3")] = 1
    m = int(jobj["M"])
    if m == 1:
        patient.iloc[0, patient.columns.get_loc("Breast - Adjusted AJCC 6th M (1988-2015)_M0")] = 1
    cs_t = int(jobj["CST"])
    if cs_t == 1:
        patient.iloc[0, patient.columns.get_loc("CS Tumor Size/Ext Eval (2004-2015)_0")] = 1
    elif cs_t == 2:
        patient.iloc[0, patient.columns.get_loc("CS Tumor Size/Ext Eval (2004-2015)_1")] = 1
    elif cs_t == 3:
        patient.iloc[0, patient.columns.get_loc("CS Tumor Size/Ext Eval (2004-2015)_3")] = 1
    elif cs_t == 4:
        patient.iloc[0, patient.columns.get_loc("CS Tumor Size/Ext Eval (2004-2015)_5")] = 1
    elif cs_t == 5:
        patient.iloc[0, patient.columns.get_loc("CS Tumor Size/Ext Eval (2004-2015)_6")] = 1
    cs_n = int(jobj["CSN"])
    if cs_n == 1:
        patient.iloc[0, patient.columns.get_loc("CS Reg Node Eval (2004-2015)_0")] = 1
    elif cs_n == 2:
        patient.iloc[0, patient.columns.get_loc("CS Reg Node Eval (2004-2015)_1")] = 1
    elif cs_n == 3:
        patient.iloc[0, patient.columns.get_loc("CS Reg Node Eval (2004-2015)_2")] = 1
    elif cs_n == 4:
        patient.iloc[0, patient.columns.get_loc("CS Reg Node Eval (2004-2015)_3")] = 1
    elif cs_n == 5:
        patient.iloc[0, patient.columns.get_loc("CS Reg Node Eval (2004-2015)_5")] = 1
    elif cs_n == 6:
        patient.iloc[0, patient.columns.get_loc("CS Reg Node Eval (2004-2015)_6")] = 1
    cs_m = int(jobj["CSM"])
    if cs_m == 1:
        patient.iloc[0, patient.columns.get_loc("CS Mets Eval (2004-2015)_0")] = 1
    elif cs_m == 2:
        patient.iloc[0, patient.columns.get_loc("CS Mets Eval (2004-2015)_1")] = 1
    elif cs_m == 3:
        patient.iloc[0, patient.columns.get_loc("CS Mets Eval (2004-2015)_3")] = 1
    elif cs_m == 4:
        patient.iloc[0, patient.columns.get_loc("CS Mets Eval (2004-2015)_5")] = 1
    elif cs_m == 5:
        patient.iloc[0, patient.columns.get_loc("CS Mets Eval (2004-2015)_6")] = 1
    laterality = int(jobj["laterality"])
    if laterality == 1:
        patient.iloc[0, patient.columns.get_loc("Laterality_Bilateral, single primary")] = 1
    elif laterality == 2:
        patient.iloc[0, patient.columns.get_loc("Laterality_Left - origin of primary")] = 1
    elif laterality == 3:
        patient.iloc[0, patient.columns.get_loc("Laterality_Only one side - side unspecified")] = 1
    elif laterality == 4:
        patient.iloc[0, patient.columns.get_loc("Laterality_Paired site, but no information concerning laterality")] = 1
    elif laterality == 5:
        patient.iloc[0, patient.columns.get_loc("Laterality_Right - origin of primary")] = 1
    breast_subtype = int(jobj["subtype"])
    if breast_subtype == 1:
        patient.iloc[0, patient.columns.get_loc("Breast Subtype (2010+)_HR+/HER2+ (Luminal B)")] = 1
    elif breast_subtype == 2:
        patient.iloc[0, patient.columns.get_loc("Breast Subtype (2010+)_HR+/HER2- (Luminal A)")] = 1
    elif breast_subtype == 3:
        patient.iloc[0, patient.columns.get_loc("Breast Subtype (2010+)_HR-/HER2+ (HER2 enriched)")] = 1
    elif breast_subtype == 4:
        patient.iloc[0, patient.columns.get_loc("Breast Subtype (2010+)_HR-/HER2- (Triple Negative)")] = 1
    radiation_recode = int(jobj["radiation_recode"])
    if radiation_recode == 1:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_Beam radiation")] = 1
    elif radiation_recode == 2:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_Combination of beam with implants or isotopes")] = 1
    elif radiation_recode == 3:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_None/Unknown")] = 1
    elif radiation_recode == 4:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_Radiation, NOS  method or source not specified")] = 1
    elif radiation_recode == 5:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_Radioactive implants (includes brachytherapy) (1988+)")] = 1
    elif radiation_recode == 6:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_Radioisotopes (1988+)")] = 1
    elif radiation_recode == 7:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_Recommended, unknown if administered")] = 1
    elif radiation_recode == 8:
        patient.iloc[0, patient.columns.get_loc("Radiation recode_Refused (1988+)")] = 1
    chemotherapy_recode = int(jobj["chemotherapy_recode"])
    if chemotherapy_recode == 1:
        patient.iloc[0, patient.columns.get_loc("Chemotherapy recode (yes, no/unk)_No/Unknown")] = 1
    elif chemotherapy_recode == 2:
        patient.iloc[0, patient.columns.get_loc("Chemotherapy recode (yes, no/unk)_Yes")] = 1


    patient = pd.concat([patient] * 10, ignore_index=False)
    patient.iloc[0, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Bilateral mastectomy")] = 1
    patient.iloc[1, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Extended radical mastectomy")] = 1
    patient.iloc[2, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Local tumor destruction")] = 1
    patient.iloc[3, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Mastectomy")] = 1
    patient.iloc[4, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Modified radical mastectomy")] = 1
    patient.iloc[5, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_None")] = 1
    patient.iloc[6, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Partial mastectomy")] = 1
    patient.iloc[7, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Radical mastectomy")] = 1
    patient.iloc[8, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Subcutaneous mastectomy")] = 1
    patient.iloc[9, patient.columns.get_loc("RX Summ--Surg Prim Site (1998+)_Total (simple) mastectomy")] = 1

    # table
    pre_haz_rate = model.predict(patient)
    haz_rate_list = pre_haz_rate.tolist()
    expg = np.exp(pre_haz_rate).reshape(1, -1)
    base_cum_haz_test = pd.DataFrame(base_cum_haz.values.reshape(-1, 1).dot(expg), index=base_cum_haz.index)
    # figure
    surv = np.exp(-base_cum_haz_test)
    line_series = []
    for i in range(10):
        line_series.append(surv.iloc[:, i].tolist())
    # surv.iloc[:, :10].plot()
    # plt.ylabel('S(t | x)')
    # _ = plt.xlabel('Time')
    # plt.show()
    res = {
        "rate": haz_rate_list,
        "series": line_series
    }

    return HttpResponse(json.dumps(res), content_type="application/json")

