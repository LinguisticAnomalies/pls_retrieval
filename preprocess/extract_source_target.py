import pandas as pd
import argparse

def main():
    """
    python extract_source_target.py \
        --data-path "./data/full_data_extract_elife_annals_medicine_reproductive/" \
        --data-file "test" \
        --source-name "abs_text" \
        --target-name "pls_text"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="path importing data file",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="path importing data file",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        help="path importing data file",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        help="path importing data file",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.data_path + args.data_file + '.csv')
    with open (args.data_path + args.data_file + '.target', 'w') as f:
        for item in df[args.target_name].values.tolist():
            f.write(item + '\n')
            f.write(item)
    with open (args.data_path + args.data_file + '.source', 'w') as f:
        for item in df[args.source_name].values.tolist():
            f.write(item + '\n')
            # f.write(item)

if __name__ == '__main__':
    main()