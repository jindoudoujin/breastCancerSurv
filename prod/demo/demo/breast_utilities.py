from utilities import is_integer, form_error_msg
import numpy as np


def is_valid_surgery(sur_sum):
    if sur_sum is False:
        raise ValueError(form_error_msg("Invalid parameter sur_sum."))
    return 1 == np.unique(sur_sum)


def map_breast_surg_type(code):
    '''
    this map method is based on the Surgery_Codes_Breast_2021.pdf from https://seer.cancer.gov/archive/manuals/2021/AppendixC/Surgery_Codes_Breast_2021.pdf
    TODO:unit test in the future
    '''
    if is_integer(code) is False:
        raise ValueError(form_error_msg("Invalid parameter code."))
    elif code == 0:
        return "None"
    elif code == 19:
        return "Local tumor destruction"
    elif code == 20 or code == 21 or code == 22 or code == 23 or code == 24:
        return "Partial mastectomy"
    elif code == 30:
        return "Subcutaneous mastectomy"
    elif code == 40 or code == 41 or code == 43 or code == 44 or code == 45 or code == 46 or code == 42 or code == 47 or code == 48 or code == 49 or code == 75:
        return "Total (simple) mastectomy"
    elif code == 76:
        return "Bilateral mastectomy"
    elif code == 50 or code == 51 or code == 53 or code == 54 or code == 55 or code == 56 or code == 52 or code == 57 or code == 58 or code == 59 or code == 63:
        return "Modified radical mastectomy"
    elif code == 60 or code == 61 or code == 64 or code == 65 or code == 66 or code == 67 or code == 62 or code == 68 or code == 69 or code == 73 or code == 74:
        return "Radical mastectomy"
    elif code == 70 or code == 71 or code == 72:
        return "Extended radical mastectomy"
    elif code == 80:
        return "Mastectomy"
    else:
        raise ValueError(form_error_msg("Invalid parameter code."))


def map_event_code(event):
    '''
    this map method is based on survival analysis
    TODO:unit test in the future
    '''
    if event is False:
        raise ValueError(form_error_msg("Invalid parameter event."))
    if event.lower() == "dead":
        return 1
    return 0
