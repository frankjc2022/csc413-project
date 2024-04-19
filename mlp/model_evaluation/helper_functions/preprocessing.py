import time
import json
import numpy as np
import pandas as pd
import argparse

    
# lot size
def handle_lot_size(x):
    
    if x["Front_ft"] == 0:
        val = x["Depth"]
    elif x["Depth"] == 0:
        val = x["Front_ft"]
    else:
        val = x["Front_ft"] * x["Depth"]
    
    if not val:
        return val
    
    # convert all units to Feet
    if x["Lotsz_code"] == "Feet":
        return val
    elif x["Lotsz_code"] == "Metres":
        return val * 3.28084
    elif x["Lotsz_code"] == "Acres":
        return val * 43560
    else:
        return val
    

def preprocessing(data):
    
    data = data.dropna(subset=['Sp_dol'])

    # convert fields to boolean
    #   "Den_fr": "Family Room - not NA: 47287/50026 - 94.52% - ['N', 'Y']",
    data['custom_den_fr'] = data['Den_fr'].apply(lambda x: True if x=='Y' else False)
    data['Taxes'] = data['Taxes'].apply(lambda x: None if x==0 else x)
    data['Tv'] = data['Tv'].apply(lambda x: None if x==0 else x)
    data['custom_tour_url'] = data['Tour_url'].apply(lambda x: True if type(x) == str and x.strip() else False)
    data['custom_fpl_num'] = data['Fpl_num'].apply(lambda x: True if x=='Y' else False)

    data['Lat'] = data['Lat'].apply(lambda x: None if x==0 else x)
    data['Lng'] = data['Lng'].apply(lambda x: None if x==0 else x)

    # new fields for special handle fields
    data["custom_lot_size"] = data.apply(lambda x:handle_lot_size(x), axis=1)
    
    return data


def convert_datatype(data):
    """
    This supposes split all features into each datatype, and select by user later. But we only pick
    those we may care about.
    """
    
    numerics_int_res = [
        "Photo_count",
        "Bath_tot", "Br", "Br_plus", "Rms", "Rooms_plus", "Kit_plus", "Num_kit",
        "Gar_spaces", "Park_spcs",
        # "Lp_dol",
        "Sp_dol", # target
    ]

    numerics_float_res = [
        "Lat", "Lng", 
        # "Taxes",
        "custom_lot_size",
    ]

    dates_res = ["Input_date"] # "Input_date" makes it worse, should we shuffle the data? Right now it's sorted by Cd (sold date)

    bools_res = ["custom_den_fr", "custom_tour_url", "custom_fpl_num"]

    categories_res = [
        "Comp_pts", # unique: 4
        "Constr1_out", "Constr2_out", # todo: they represent the same thing, (e.g. A,B = B,A) # unique: 14
        "Bsmt1_out", "Bsmt2_out", # todo: they represent the same thing, (e.g. A,B = B,A) # unique: 14
        "Yr_built", # ['0-5', '100+', '16-30', '31-50', '51-99', ...] unique: 7
        "Acres", # unique: 9
        "Sqft", # unique: 9
        "Style", # unique: 17
        "Type_own1_out", # unique: 17
        # "Spec_des1_out", # unique 6, though almost all are Unknown (46070 over 50026)
        "Area", # unique 7 (we restricted the records to 7 areas)
        "Municipality_district", # unique 86 (within these 7 areas)
        "Community", # unique 579 (within these 7 areas)
    ] # worse: "Sewer", "Heating",

    # features_res = numerics_float_res + numerics_int_res + bools_res + categories_res + dates_res
    # print("features:", len(features_res))

    # TODO: handle: Zip?, Input_date?

    for num in numerics_float_res:
        data[num] = data[num].fillna(0).astype(np.float)

    for num in numerics_int_res:
        data[num] = data[num].fillna(0).round().astype('int64')


    for category in categories_res:
        data[category] = data[category].astype("category")

    for d in dates_res:
        # data[d] = data[d].apply(lambda x: x.Timestamp.value)
        # data[d] = data[d].dt.strftime("%Y%m%d").astype(int)
        data[d] = data[d].str.replace("-","").fillna(0).astype(int)
        
    return data


def export_data(data, config):
    
    feature_lis = []
    for k,v in config["features"].items():
        feature_lis.extend(v)
    
    data = data[feature_lis]
    
    if "testset_percentage" in config and config["testset_percentage"]:
        data_test, data_train = np.split(data, [int(config["testset_percentage"]*len(data))])
        data_train.to_csv(config["save_path"])
        data_test.to_csv(config["save_path"].replace(".csv", "_test.csv"))
        print(f"Training size (without index): {data_train.shape}")
        print(f"Test size (without index): {data_test.shape}")
    else:
        data.to_csv(config["save_path"])
        print(f"Training size (without index): {data.shape}")
    
    # save metadata
    with open(config["save_path"].replace(".csv", ".json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, ensure_ascii=False))
        
    