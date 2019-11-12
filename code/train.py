import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)
# $$$
parser.add_argument("--config_num", type = int, default= 0, help= 'choose which config to use')
#
parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')


args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
}
# $$$
con = None
if args.config_num == 0:
	con = config.Config(args)
elif args.config_num == 1:
	con = config.Config_part1(args)
elif args.config_num == 2:
	con = config.Config_part2(args)
elif args.config_num == 3:
	con = config.Config_part3(args)
elif args.config_num == 4:
	con = config.Config_part4(args)
elif args.config_num == 5:
	con = config.Config_part5(args)
assert con

#
con = config.Config(args)
con.set_max_epoch(200)
con.load_train_data()
con.load_test_data()
# con.set_train_model()
print("args.config_num:{}".format(args.config_num))
con.train(model[args.model_name], args.save_name)
