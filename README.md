# Tutorial - Deploy DeepSeek-R1-Distill-Qwen-32B using Inferless
[DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) is a distilled variant within the DeepSeek-R1 series. The dataset used for training is meticulously curated from the DeepSeek-R1 model, with Qwen2.5-32B serving as the foundational base model. This model has undergone supervised fine-tuning to achieve enhanced performance and efficiency.

## Model acquisition, GPU and billing prerequisites

This repository is an **Inferless deployment example**, not a pretrained model distribution or training-data release. The model weights come from [DeepSeek-R1-Distill-Qwen-32B on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B); check the model card and applicable model license before serving it. No model weights or fine-tuning dataset are checked in.

**What you must provide:** an Inferless account/workspace, authorization to import this GitHub source, a supported GPU machine/region and spending allowance. The checked-in [`inferless.yaml`](inferless.yaml) config proposes a **dedicated A100** with 200 GiB host RAM, 20 vCPUs and `min_replica: 0`; these are template choices, not confirmed available products or prices. Determine the current GPU rate, minimum charge, storage and idle/scale-down charges **before deploying**.

### End-to-end deployment checklist

1. Fork or otherwise authorize this repo for your Inferless workspace.
2. Review [`app.py`](app.py), [`input_schema.py`](input_schema.py), [`inferless-runtime-config.yaml`](inferless-runtime-config.yaml) and [`inferless.yaml`](inferless.yaml). Confirm package versions and the model identifier against current provider APIs.
3. Create/select a compatible Inferless runtime and GPU in your region. Update `inferless.yaml` with the current workspace, runtime and resource settings; a blank custom-runtime ID does **not** provision the selected runtime automatically.
4. Import the Git repository as custom code, supply required secrets using the provider's secret store, and deploy.
5. Inspect provider logs for model download, tokenizer initialization and inference readiness. Use the exact endpoint and bearer token issued for **your deployment** to send the documented `curl` request.
6. Track cost and latency on your own hardware. The throughput, cold-start and token rate values below are historical example measurements, **not guarantees**.

**No separate dataset acquisition is required for inference.** Text prompts are supplied as requests. If you want to fine-tune, obtain an authorized corpus separately; training, evaluation and dataset licenses are outside this sample's scope.

## TL;DR:
- Deployment of DeepSeek-R1-Distill-Qwen-32B model using [vLLM](https://github.com/vllm-project/vllm).
- You can expect an average tokens/sec of `21.95` and a latency of `5.88` sec for generating a text of `128` tokens. This setup has an average cold start time of `39.95` sec.
- Dependencies defined in `inferless-runtime-config.yaml`.
- GitHub/GitLab template creation with `app.py`, `inferless-runtime-config.yaml` and `inferless.yaml`.
- `InferlessPythonModel` class in `app.py` with `initialize`, `infer`, and `finalize` functions.
- Custom runtime creation with necessary system and Python packages.
- Recommended GPU: NVIDIA A100 for optimal performance.
- Custom runtime selection in advanced configuration.
- Final review and deployment on the Inferless platform.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the `Add a custom model` button.

- Select `Github` as the method of upload from the Provider list and then select your Github Repository and the branch.
- Choose the type of machine, and specify the minimum and maximum number of replicas for deploying your model.
- Configure Custom Runtime ( If you have pip or apt packages), choose Volume, Secrets and set Environment variables like Inference Timeout / Container Concurrency / Scale Down Timeout
- Once you click “Continue,” click Deploy to start the model import process.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer <your_api_key>' \
    --data '{
    "inputs": [
        {
            "name": "prompt",
            "shape": [
                1
            ],
            "data": [
                "Explain Deep Learning."
            ],
            "datatype": "BYTES"
        },
        {
            "name": "temperature",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                0.7
            ],
            "datatype": "FP64"
        },
        {
            "name": "top_p",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                0.1
            ],
            "datatype": "FP64"
        },
        {
            "name": "repetition_penalty",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                1.18
            ],
            "datatype": "FP64"
        },
        {
            "name": "top_k",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                40
            ],
            "datatype": "INT32"
        },
        {
            "name": "max_tokens",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                256
            ],
            "datatype": "INT32"
        }
    ]
}'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. The `InferlessPythonModel` has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The infer function leverages both RequestObjects and ResponseObjects to handle inputs and outputs in a structured and maintainable way.
- RequestObjects: Defines the input schema, validating and parsing the input data.
- ResponseObjects: Encapsulates the output data, ensuring consistent and structured API responses.

```python
  def infer(self, request: RequestObjects) -> ResponseObjects:
    sampling_params = SamplingParams(temperature=request.temperature,top_p=request.top_p,repetition_penalty=request.repetition_penalty,
                     top_k=request.top_k,max_tokens=request.max_tokens)
    input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": request.prompt}], tokenize=False)
    result = self.llm.generate(input_text, sampling_params)
    result_output = [output.outputs[0].text for output in result]

    generateObject = ResponseObjects(generated_text = result_output[0])        
    return generateObject
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.
```python
def finalize(self):
    self.llm = None
```


For more information refer to the [Inferless docs](https://docs.inferless.com/).
