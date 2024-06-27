# Setup Guide

## Create Cluster
We export a few environment variables before proceeding:

```bash
export CLUSTER_NAME=****
export REGION=us-west-2
export ACCOUNT_ID==****
export CLUSTER_REGION=us-west-2
```
>Note: For this deployment, a VPC (with public subnets only) is created manually and then used for the cluster.

```bash
envsubst < 00_cluster.yaml | eksctl create cluster -f -
```

## Install the Metrics API

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

## Install AWS IAM Authenticator
```bash
curl -Lo aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.5.9/aws-iam-authenticator_0.5.9_linux_amd64
chmod +x ./aws-iam-authenticator
sudo cp ./aws-iam-authenticator /usr/local/bin
```


## Setup EBS Access

### Create IRSA for EBS

```bash
eksctl create iamserviceaccount \
  --name ebs-csi-controller-sa \
  --namespace kube-system \
  --cluster ${CLUSTER_NAME} \
  --role-name AmazonEKS_EBS_CSI_DriverRole \
  --role-only \
  --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
  --approve \
  --region ${REGION}
```

### Create the EBS CSI Driver Addon


```bash
eksctl create addon --name aws-ebs-csi-driver --cluster ${CLUSTER_NAME} --service-account-role-arn arn:aws:iam::${ACCOUNT_ID}:role/AmazonEKS_EBS_CSI_DriverRole --region ${REGION} --force

```


## Clone Kubeflow Repos

```bash
export KUBEFLOW_RELEASE_VERSION=v1.7.0
export AWS_RELEASE_VERSION=v1.7.0-aws-b1.0.3
git clone https://github.com/awslabs/kubeflow-manifests.git && cd kubeflow-manifests
git checkout ${AWS_RELEASE_VERSION}
git clone --branch ${KUBEFLOW_RELEASE_VERSION} https://github.com/kubeflow/manifests.git upstream
```

edit the following section in `Makefile`

```
install-helm: 
	wget <https://get.helm.sh/helm-v3.12.2-linux-amd64.tar.gz>
	tar -zxvf helm-v3.12.2-linux-amd64.tar.gz
	sudo mv linux-amd64/helm /usr/local/bin/helm
	helm version
```

## Install Pre-reqs for Kubeflow
```bash
make install-tools
```


## Install Kubeflow
```bash
make deploy-kubeflow INSTALLATION_OPTION=helm DEPLOYMENT_OPTION=vanilla
```

## Enable Minio Access

```bash
kubectl apply -f 01_minio_secret.yaml -n kubeflow-user-example-com
```

## ConfigMap Patch
We update the Knative domain that is used for the KServe routes.

Example guides including the ones [here](https://knative.dev/docs/serving/using-a-custom-domain/#configuring-the-default-domain-for-all-knative-services-on-a-cluster) and [here](https://ibm.github.io/manifests/docs/deployment/authentication/#optional---kserve-configuration)

```bash
kubectl patch cm config-domain --patch '{"data":{"emlo.mmg":""}}' -n knative-serving
```
