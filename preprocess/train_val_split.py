import random
import os
import math
import pandas as pd
import argparse

def split_train_val(input_list, num_val=0.8):
    n = math.floor(len(input_list) * num_val)
    trainlist = random.sample(range(0, len(input_list)), n)
    wholelist = [i for i in range(len(input_list))]
    vallist = list(set(wholelist) - set(trainlist))
    return trainlist, vallist

def main():
    """
    Usage::
    
        python train_val_split.py \
        --data-path ../MedLane_data/
        --save-path ../MedLane_data/MedLane_pretrain_data/
        --data-file MedLane_pretrain_data.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="path importing data file",
    )
    
    parser.add_argument(
        "--save-path",
        required=True,
        type=str,
        default="test(2030)_new.txt",
        help="path exporting data file",
    )

    parser.add_argument(
        "--data-file",
        required=True,
        type=str,
        default="full_data.csv",
        help="the full data",
    )

    
    args = parser.parse_args()
    
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    
    df = pd.read_csv(args.data_path + args.data_file)

    train_val_list, test_list = split_train_val(df, 0.9)
    train_val = df.iloc[train_val_list]
    test = df.loc[test_list]
    train_list, val_list = split_train_val(train_val, 0.8)
    train = train_val.iloc[train_list]
    val = train_val.iloc[val_list]
    print('train:', len(train))
    print('val:', len(val))
    print('test:', len(test))

    train.to_csv(args.save_path + "train.csv", index=False)
    val.to_csv(args.save_path + "val.csv", index=False)
    test.to_csv(args.save_path + "test.csv", index=False)

if __name__ ==  "__main__":
    main()