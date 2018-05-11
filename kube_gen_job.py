import yaml
import copy
import argparse
import random
import os

pserver = {
    "apiVersion": "extensions/v1beta1",
    "kind": "ReplicaSet",
    "metadata": {
        "name": "jobname-pserver"
    },
    "spec": {
        "replicas": 1,
        "template" : {
            "metadata": {
                "labels": {
                    "paddle-job-pserver": "jobname"
                }
            },
            "spec": {
                "hostNetwork": True,
                "imagePullSecrets": [{
                    "name": "job-registry-secret"
                }],
                "containers": [{
                    "name": "pserver",
                    "image": "",
                    "imagePullPolicy": "Always",
                    "ports": [
                        {
                            "name": "jobport-1",
                            "containerPort": 1
                        }
                    ],
                    "env": [],
                    "command": ["paddle_k8s", "start_pserver"],
                    "resources": {
                        "requests": {
                            "memory": "10Gi",
                            "cpu": "4"
                        },
                        "limits": {
                            "memory": "10Gi",
                            "cpu": "4"
                        }
                    }
                }]
            }
        }
    }
}

trainer = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {
        "name": "jobname-pserver"
    },
    "spec": {
        "parallelism": 4,
        "completions": 4,
        "template" : {
            "metadata": {
                "labels": {
                    "paddle-job": "jobname"
                }
            },
            "spec": {
                "hostNetwork": True,
                "imagePullSecrets": [{
                    "name": "job-registry-secret"
                }],
                "restartPolicy": "Never",
                "containers": [{
                    "name": "trainer",
                    "image": "",
                    "imagePullPolicy": "Always",
                    "ports": [
                        {
                            "name": "jobport-1",
                            "containerPort": 1
                        }
                    ],
                    "env": [],
                    "command": ["paddle_k8s", "start_trainer", "v2"],
                    "resources": {
                        "requests": {
                            "memory": "10Gi",
                            "cpu": "4",
                        },
                        "limits": {
                            "memory": "10Gi",
                            "cpu": "4",
                        }
                    }
                }]
            }
        }
    }
}

envs = [
    # {"name": "PADDLE_JOB_NAME", "value": ""},
    # {"name": "TRAINERS", "value": "4"},
    # {"name": "PSERVERS", "value": "4"},
    # {"name": "ENTRY", "value": ""},
    # {"name": "PADDLE_INIT_PORT", "value": ""},
    # envs that don't need to change
    {"name": "TOPOLOGY", "value": ""},
    {"name": "TRAINER_PACKAGE", "value": "/workspace"},
    {"name": "PADDLE_INIT_NICS", "value": "eth2"},
    {"name": "LD_LIBRARY_PATH", "value": "/usr/local/lib:/usr/local/nvidia/lib64"},
    {"name": "NAMESPACE", "valueFrom": {
        "fieldRef": {
            "fieldPath": "metadata.namespace"
        }
    }}
]

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dist job yamls.')
    
    parser.add_argument('--jobname', default="paddlejob",
                        help='unique job name')
    parser.add_argument('--cpu', default=1,
                        help='CPU cores per trainer node')
    parser.add_argument('--pscpu', default=1,
                        help='CPU cores per pserver node')
    parser.add_argument('--gpu', default=0,
                        help='num of GPUs per node')
    parser.add_argument('--image', default="bootstrapper:5000/fluid_benchmark:gpu",
                        help='num of GPUs per node')
    parser.add_argument('--pservers', default=1,
                        help='num of pservers')
    parser.add_argument('--trainers', default=1,
                        help='num of trainers')
    parser.add_argument('--memory', default=1,
                        help='trainer memory')
    parser.add_argument('--psmemory', default=1,
                        help='pserver memory')
    parser.add_argument('--port', default=30236,
                        help='num of trainers')
    parser.add_argument('--entry', default="python train.py",
                        help='command to run')
    parser.add_argument('--fluid', default=1,
                        help='whether is fluid job')

    args = parser.parse_args()
    return args

def gen_job():
    ps = pserver
    tn = trainer
    args = parse_args()

    ps_container = ps["spec"]["template"]["spec"]["containers"][0]
    tn_container = tn["spec"]["template"]["spec"]["containers"][0]

    if args.fluid:
        ps_container["command"] = \
            ["paddle_k8s", "start_fluid"]
        tn_container["command"] = \
            ["paddle_k8s", "start_fluid"]
    ps["metadata"]["name"] = args.jobname + "-pserver"
    ps["spec"]["template"]["metadata"]["labels"]["paddle-job-pserver"] = args.jobname
    tn["metadata"]["name"] = args.jobname + "-trainer"
    tn["spec"]["template"]["metadata"]["labels"]["paddle-job"] = args.jobname
    

    ps_container["image"] = args.image
    tn_container["image"] = args.image

    ps_container["resources"]["requests"]["cpu"] = str(args.pscpu)
    ps_container["resources"]["requests"]["memory"] = str(args.psmemory) + "Gi"
    ps_container["resources"]["limits"]["cpu"] = str(args.pscpu)
    ps_container["resources"]["limits"]["memory"] = str(args.psmemory) + "Gi"

    tn_container["resources"]["requests"]["cpu"] = str(args.cpu)
    tn_container["resources"]["requests"]["memory"] = str(args.memory) + "Gi"
    tn_container["resources"]["limits"]["cpu"] = str(args.cpu)
    tn_container["resources"]["limits"]["memory"] = str(args.memory) + "Gi"
    if args.gpu > 0:
        tn_container["resources"]["requests"]["alpha.kubernetes.io/nvidia-gpu"] = str(args.gpu)
        tn_container["resources"]["limits"]["alpha.kubernetes.io/nvidia-gpu"] = str(args.gpu)

    ps["spec"]["replicas"] = args.pservers
    tn["spec"]["parallelism"] = args.trainers
    tn["spec"]["completions"] = args.trainers
    ps_container["ports"][0]["name"] = "jobport-" + str(args.port)
    ps_container["ports"][0]["containerPort"] = args.port
    spreadport = random.randint(40000, 60000)
    tn_container["ports"][0]["name"] = "spr-" + str(spreadport)
    tn_container["ports"][0]["containerPort"] = spreadport


    # {"name": "PADDLE_JOB_NAME", "value": ""},
    # {"name": "TRAINERS", "value": "4"},
    # {"name": "PSERVERS", "value": "4"},
    # {"name": "ENTRY", "value": ""},
    # {"name": "PADDLE_INIT_PORT", "value": ""},
    envs.append({"name": "PADDLE_JOB_NAME", "value": args.jobname})
    envs.append({"name": "TRAINERS", "value": str(args.trainers)})
    envs.append({"name": "PSERVERS", "value": str(args.pservers)})
    envs.append({"name": "ENTRY", "value": args.entry})
    envs.append({"name": "PADDLE_INIT_PORT", "value": str(args.port)})

    volumes = [
        {
            "name": "nvidia-driver",
            "hostPath": {"path": "/usr/local/nvidia/lib64"}
        }
    ]
    volumeMounts = [
        {
            "mountPath": "/usr/local/nvidia/lib64",
            "name": "nvidia-driver"
        }
    ]
    tn["spec"]["template"]["spec"]["volumes"] = volumes
    tn_container["volumeMounts"] = volumeMounts

    ps_container["env"] = envs
    tn_container["env"] = envs

    os.mkdir(args.jobname)
    with open("%s/pserver.yaml" % args.jobname, "w") as fn:
        yaml.dump(ps, fn)
    with open("%s/trainer.yaml" % args.jobname, "w") as fn:
        yaml.dump(tn, fn)

if __name__ == "__main__":
    gen_job()