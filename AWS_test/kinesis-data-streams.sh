# !/bin/bash

# get the AWS CLI version
aws --version

# PRODUCER
# CLI v2
aws kinesis put-record --stream-name test
--partition-key user1 --data "user signup"
--cli-binary-key user1 --data "user signup"

# CLI v1
aws kinesis put-recod --stream-name test
--partition-key user1 --data "user signup"

# CONSUMER

# describe the stream
aws kinesis describe-stream --stream-name test

# Consume some data
aws kinesis get-shard-iterator --stream-name test
--shard-id shardId-0000000000
--shard-iterator-type TRIM_HORIZON

aws kinesis get-records --shard-iterator <>