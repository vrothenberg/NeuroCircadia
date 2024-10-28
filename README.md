# NeuroCircadia


**Chainlit** 
`pip install chainlit`

**Langserve**
`pip install "langserve[all]"`

**Google** 
`conda install -c conda-forge google-cloud-sdk`

`pip install google-cloud-aiplatform`

Add service account keys to your ~/.bashrc or ~/.zshrc file:

`export GOOGLE_APPLICATION_CREDENTIALS="/path/to/rbio-p-datasharing-b5c1d9a2deba.json"`

`pip install langchain langchain-core langchain-community langchain-google-vertexai`


## Run Chainlit server

In `NeuroCircadia/chainlit-app` run `chainlit run app.py -w`

