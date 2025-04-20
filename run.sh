set -x

input_file=$1
output_path=$2
python infer.py --dataset_dir="$input_file" --output_path="$output_path"