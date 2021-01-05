#! /bin/bash

DW_AUTH_TOKEN=`gcloud secrets versions access --secret=data_world_api_token latest`

pipenv lock -r > requirements.txt
gcloud functions deploy process-crdt-data --entry-point run_process_data --runtime python37 \
    --trigger-topic process-crdt-data \
    --memory 1024MB \
    --max-instances 1 \
    --update-env-vars="DW_AUTH_TOKEN=${DW_AUTH_TOKEN}" \
    --timeout 540
