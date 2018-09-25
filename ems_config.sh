#!/bin/bash

# move sensitive files back into view of the EMS
cp ./data/master_participants_september.json ./data/mperf_ids.txt ./data/user_durations.pickle ./data/user_ids.py ./data/userid_map.py ./data/master_streams_aws.json .

# VM
if [[ $(hostname -s) = cerebralcortexems ]]; then

    echo VM DETECTED

    export PYSPARK_PYTHON=/usr/bin/python3.6
    export SPARK_HOME=/usr/local/spark/
    export PATH=$SPARK_HOME/bin:$PATH
    export SPARK_MASTER="local[*]"

    export PYSPARK_DRIVER_PYTHON=/usr/bin/python3.6
    export PYTHONPATH=/usr/local/spark/python/lib/py4j-0.10.4-src.zip:/usr/local/spark/python/

    sudo chmod 777 /tmp/logs/spark.log
    sudo mkdir /var/log/cerebralcortex/
    sudo chmod 777 /var/log/cerebralcortex/

# aws
elif [[ $(hostname -s) = ip-172-31-23-188 ]]; then

    echo AWS DETECTED

    export PYSPARK_PYTHON=/usr/bin/python3.6
    export SPARK_HOME=/usr/local/spark/
    export PATH=$SPARK_HOME/bin:$PATH
    export SPARK_MASTER="local[*]"

    export PYSPARK_DRIVER_PYTHON=/usr/bin/python3.6
    export PYTHONPATH=/usr/local/spark/python/lib/py4j-0.10.4-src.zip:/usr/local/spark/python/

    sudo chmod 777 /tmp/logs/spark.log
    sudo mkdir /var/log/cerebralcortex/
    sudo chmod 777 /var/log/cerebralcortex/

# star wars
elif [[ $(hostname -s) = *10dot* ]] || [[ $(hostname -s) = *memphis* ]] || [[ $(hostname -s) = "dagobah" ]]; then

    echo PRODUCTION DETECTED
    
    export SPARK_MASTER="local[4]"

    export PYSPARK_DRIVER_PYTHON=/usr/bin/python3.6
    export PYTHONPATH=/usr/local/spark/python/lib/py4j-0.10.4-src.zip:/usr/local/spark/python/:$PYTHONPATH
    export PYTHONPATH=/cerebralcortex/code/CerebralCortex/:$PYTHONPATH
    export PYTHONPATH=/cerebralcortex/code/ems/EMS/:$PYTHONPATH

else
    echo unknown environment

fi
