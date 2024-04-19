import time
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
# for visualizing missing values
import missingno as msno
# from IPython.display import display
import dataframe_image as dfi

# for display plot inline
# %matplotlib inline
# change the style
matplotlib.style.use('ggplot')

from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    
def eval_metrics(model_pred, target):
    
    mae = mean_absolute_error(model_pred, target)
    mse = mean_squared_error(model_pred, target, squared=True)

    x = {
        "pred": model_pred,
        "pred_y": target
    }
    perc = np.abs(model_pred - target)/target
    median = perc.median()
    count_5 = perc[perc <= 0.05].count() / model_pred.size if model_pred.size else 0
    count_10 = perc[perc <= 0.10].count() / model_pred.size if model_pred.size else 0
    count_20 = perc[perc <= 0.20].count() / model_pred.size if model_pred.size else 0
    
    # print(f"{mae:.2f}")
    # # print(f"{mse:.2f}")
    # print(f"{np.sqrt(mse):.2f}")
    # print("median:", median)
    # print("<= 5%:", count_5)
    # print("<= 10%:", count_10)
    # print("<= 20%:", count_20)
    
    return {
        "mae": mae,
        "mse": mse, 
        "rmse": np.sqrt(mse),
        "median": median,
        "count_5": count_5,
        "count_10": count_10,
        "count_20": count_20
    }

def split_data(data, valid_perc=None, test_perc=None):
    """
    Using np.split:
    https://numpy.org/doc/stable/reference/generated/numpy.split.html
    Assuming the dataset is sort by sold date desc. Using the latest for test, then for validation.
    """
    
    if test_perc and valid_perc:
        data_test, data_validate, data_train = np.split(data, [int(test_perc*len(data)), int((valid_perc+test_perc)*len(data))])
    elif valid_perc:
        data_validate, data_train = np.split(data, [int(valid_perc*len(data))])
        data_test = None
    else:
        data_train = data
        data_validate = None
        data_test = None
        

    data_train_x = data_train.drop("Sp_dol", axis=1)
    data_train_y = data_train["Sp_dol"]
    
    if valid_perc:
        data_validate_x = data_validate.drop("Sp_dol", axis=1)
        data_validate_y = data_validate["Sp_dol"]
    
    if test_perc:
        data_test_x = data_test.drop("Sp_dol", axis=1)
        data_test_y = data_test["Sp_dol"]

    
    print("     all data:", data.shape)
    print(   "train data:", data_train.shape)
    
    if valid_perc:
        print("validate data:", data_validate.shape)
    
    if test_perc:
        print("    test data:", data_test.shape)
    
    result = {
        # "train": data_train,
        "train_x": data_train_x,
        "train_y": data_train_y
    }
    
    if valid_perc:
        result.update({
            # "validate": data_validate,
            "validate_x": data_validate_x,
            "validate_y": data_validate_y,
        })
    
    if test_perc:
        result.update({
            # "test": data_test,
            "test_x": data_test_x,
            "test_y": data_test_y,
        })
    
    
    return result


def predict_result(data, target, predict, save_path=None):
    """
    example usage:
    result_res = predict_result(model_res, split_data_res["validate_x"], split_data_res["validate_y"], "2023_res_validate_result.csv")
    """
    
    # pred = my_model.predict(d_x)
    d_x = data
    pred = predict
    d_y = target

    try:
        d_x_index = d_x.index
    except:
        d_x_index = list(range(d_x.shape[0]))

    output = pd.DataFrame({'Ml_num': d_x_index, 'predict': pred, "actual":d_y, "diff": pred - d_y, "diff_perc": round(np.abs(pred - d_y)/d_y, 4)})
    
    if save_path:
        output.to_csv(save_path, index=True)
    
    return output


def display_worst_prediction(data, target, predict, name="", topk=10):
    
    # pred = my_model.predict(d_x)
    # pred_y = d_y
    d_x = data
    pred = predict
    pred_y = target

    try:
        d_x_index = d_x.index
    except:
        d_x_index = list(range(d_x.shape[0]))
    
    pred = pd.DataFrame({"pred":pred, "Ml_num":d_x_index, "pred_y":pred_y})
    pred.set_index("Ml_num", inplace=True)
    
    pred["diff"] = np.abs(pred["pred"]-pred["pred_y"]) / pred["pred_y"]
    
    res = pred[["pred", "pred_y", "diff"]].sort_values(by=["diff"],ascending=False) \
            .head(topk).rename(columns={"pred": "Prediction", "pred_y": "Sale Price", "diff":"Different Percentage"}) \
            .style.format({"Prediction":"{:,.0f}", "Sale Price":"{:,.0f}", "Different Percentage":"{:,.2%}"}) \
            .set_table_styles([{
                 'selector': 'caption',
                 'props': 'font-weight:bold;font-size:1.25em;'
             }], overwrite=False) \
            .set_caption(f"Top {topk} Worst Predict Result of Listings" + ("" if not name else f"<br>{name}"))
            # .set_caption(f"Worst Predict Result of Listings<br>({data_name} set)"))


    # print("Total predicted:", pred.shape)
    # print("Total predicted with difference > 0.5:", pred[pred["diff"]>0.5].shape)
    
    return res

    
def display_predict_result(data, target, predict, name="", group_by="Area", sort_by="Homes", ascending=False):

    
    # pred = my_model.predict(d_x)
    # pred_y = d_y
    d_x = data
    pred = predict
    pred_y = target

    # group_by = "Area" #"Area" # "Municipality_district" # "S_r"
    # sort_by = "Homes" # "Municipality_district" # "Homes"
    # ascending = False

    try:
        d_x_index = d_x.index
    except:
        d_x_index = list(range(d_x.shape[0]))
    
    pred = pd.DataFrame({"pred":pred, "Ml_num":d_x_index, "pred_y":pred_y})
    pred = pd.concat([pred, d_x[[group_by]]], axis=1, join='inner')
    pred.set_index("Ml_num", inplace=True)

    
    def calculation(x):
        perc = np.abs(x["pred"] - x["pred_y"])/x["pred_y"]
        median = perc.median()
        count_5 = perc[perc <= 0.05].count() / x["pred"].size if x["pred"].size else 0
        count_10 = perc[perc <= 0.10].count() / x["pred"].size if x["pred"].size else 0
        count_20 = perc[perc <= 0.20].count() / x["pred"].size if x["pred"].size else 0

        # print(x.shape) <= (15, 3)
        # print(x.size) <= 45
        # print(x.size.astype(int)) <= 45

        res = {'Median Error': median, 'Within 5% of Sales Price': count_5, 'Within 10% of Sales Price': count_10, 'Within 20% of Sales Price': count_20, "Homes":x["pred"].size}
        return pd.Series(res, index=res.keys())


    # TODO: why groupby contains empty dataframe, a workaround right now is prevent zero division in calculation()
    result = pred.groupby([group_by]).apply(calculation)
    result.loc["All Areas"] = calculation(pred)

    result = result.sort_values(by=sort_by, ascending=ascending)
    result_style = result.style.format({'Median Error': "{:.2%}",'Within 5% of Sales Price': "{:.2%}",'Within 10% of Sales Price': "{:.2%}",'Within 20% of Sales Price': "{:.2%}",'Homes': "{:,.0f}"}) \
                    .set_table_styles([{
                                 'selector': 'caption',
                                 'props': 'font-weight:bold;font-size:1.25em;'
                             }], overwrite=False) \
                    .set_caption(name)
    # display(result_style)
    # result[['Median Error','Within 5% of Sales Price','Within 10% of Sales Price','Within 20% of Sales Price']] = result[['Median Error','Within 5% of Sales Price','Within 10% of Sales Price','Within 20% of Sales Price']].applymap('{:.2%}'.format)
    # result['Homes'] = result['Homes'].apply('{:,.0f}'.format)
    # result.dfi.export(f"2023_res_validate_result.png")
    
    return result_style