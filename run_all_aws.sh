data_file=$1

#Run IGTB 
bash run_in_serial_aws.sh $data_file data/experiment_definitions_aws/initial/

#Run DGTB
bash run_in_serial_aws.sh $data_file data/experiment_definitions_aws/daily/

#Run CSV generation
python3.6 standalone_csv_generator.py --data-directory experiment_output/clipped_summary/ --experiment-name may14