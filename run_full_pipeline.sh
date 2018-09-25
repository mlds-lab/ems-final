data_dir=master_data_file

. ./ems_config.sh

# summarize data
echo "Summarizing data to directory $data_dir"

# sh run_master_summarize.sh        # uncomment for production run!!
sh run_master_summarize.sh test     # comment for production run!!

# clip summary dataframe
echo "Clipping data in directory $data_dir "
python3 clip_summary.py --data-directory $data_dir --days 70 --verbose

# run experiments in EMS using clipped summary and produce CSVs
echo "Starting EMS and results generator"
# sh run_all_aws.sh 

# run IGTB 
bash run_in_serial_aws.sh clipped_summary.pkl data/experiment_definitions_aws/initial/

# run DGTB
bash run_in_serial_aws.sh clipped_summary.pkl data/experiment_definitions_aws/daily/

# run CSV generation
python3.6 standalone_csv_generator.py --data-directory experiment_output/clipped_summary/ --experiment-name may14