import argparse
from continued_training import get_local_parser,main


setting_names= ['beta','learning_rate','num_views','renderer','query_array','init','contrast_lambda','temp','use_zero_conv','background']

settings = [[200],
            [01e-05],
            [3],
            ['nvr+'],
            ["easy_3"],
            ["og_init"],
            [0.01],
            [50],
            [False,True],
            ['default']]


num_settings = 1
for sett in settings:
    num_settings *= len(sett)

args = get_local_parser()


def get_setting(setting_number, total, settings, setting_names):
    indexes = []  ## this says which hyperparameter we use
    remainder = setting_number
    for setting in settings:
        division = int(total / len(setting))
        index = int(remainder / division)
        remainder = remainder % division
        indexes.append(index)
        total = division
    actual_setting = {}
    for j in range(len(indexes)):
        actual_setting[setting_names[j]] = settings[j][indexes[j]]
    return indexes, actual_setting


_,params = get_setting(args.setting,num_settings,settings,setting_names)

for k,v in params.items():
    print(k,v)
    setattr(args,k,v)

main(args)