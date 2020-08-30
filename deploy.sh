#! /bin/bash

pipenv lock -r > requirements.txt
gcloud functions deploy update-crdt-data --entry-point run_process_data --runtime python38 --trigger-http --allow-unauthenticated
