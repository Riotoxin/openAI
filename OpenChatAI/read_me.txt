# Preparing Datasets, Fine tune Models & using the Fine Tuned Models
openai tools fine_tunes.prepare_data -f .\cities.csv
openai --api-key OPENAI_API_KEY  api fine_tunes.create -t .\cities_prepared.jsonl -m davinci
openai --api-key OPENAI_API_KEY api completions.create -m davinci:ft-personal-2023-01-03-08-52-21 -p "What is the Population Growth of Doral?"
openai --api-key OPENAI_API_KEY api completions.create -m davinci:ft-personal-2023-02-02-17-21-01 -p "What is the Budget of Tenant 7?"

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cpu.html

Steps:
1) Use csvGeneration API to generate CSV file.
2) Follow the above instructions to prepare data, finetune model & use the finetuned model.

1) Way to Download the Model. - Data is not uploaded.
2) How & what is the infra required to train the model
3) ML-OPS.
4) Limitations of the model & Training needs - Needs Prompt & Completion modeled data
5) What are the Data Formats required to Train the Model - JSONL, CSV, JSON, TSV
