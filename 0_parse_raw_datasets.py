import argparse
import json
import os

from image2source.sxn_parser import encode_sxn_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sourceDir", help="the source directory of the parsed data", default="datasets/raw", type=str)
    parser.add_argument(
        "--targetDir", help="the target directory of the parsed data", default="datasets/parsed", type=str)
    args = parser.parse_args()

    dataset_dict = {}
    for website_directory in [
        f for f in os.listdir(args.sourceDir) if not os.path.isfile(os.path.join(args.sourceDir, f))]:
        dataset_dict.update(
            encode_sxn_folder(os.path.join(args.sourceDir, website_directory), args.targetDir))

    # write the dataset dict to a file
    with open(os.path.join(args.targetDir, 'website_stats.json'), 'w') as f:
        json.dump(dataset_dict, f, indent=2)
