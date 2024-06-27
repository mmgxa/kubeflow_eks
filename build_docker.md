
# Instructions for Docker Images

The code is to be executed inside the relevant subdirectories.
## Notebook
The original image is located [here](https://github.com/awslabs/kubeflow-manifests/blob/release-v1.7.0-aws-b1.0.3/components/notebook-dockerfiles/pytorch/cpu.Dockerfile).

```bash
docker build -t kubeflow_aws_p310-pt21_cpu:notebook .
```

## Tasks
The original image is located [here](https://github.com/kubeflow/pipelines/blob/master/samples/contrib/pytorch-samples/Dockerfile).

This image is used for the following steps:
- preprocessing
- training
- minio

```bash
docker build -t kubeflow_aws_p310-pt21_cpu:tasks .
```

Due to outdated code in the `pytorch_kfp_components` library - specifically the trainer, manual editing was done inside the Docker file.

## Serve

The original image is located [here](https://github.com/kubeflow/pipelines/blob/master/components/kserve/Dockerfile).

Note that the actual image being used is listed [here](https://quay.io/repository/aipipeline/kserve-component?tab=tags&tag=v0.11.1)

```bash
docker build -t kubeflow_aws_p310-pt21_cpu:serve .
```
