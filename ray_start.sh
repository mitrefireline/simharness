#! /bin/bash

# For the documentation of `ray start`, see
# https://docs.ray.io/en/latest/cluster/cli.html#ray-start


function usage() {
    cat <<USAGE

    Usage: $0 [OPTIONS]

    Options:
        -o, --object-store-memory INTEGER   
                                        The amount of memory (in bytes) to start the
                                        object store with. By default, this is
                                        capped at 20GB but can be set higher.
        -c, --num-cpus INTEGER          the number of CPUs on this node
        -G, --visible-devices TEXT      the list of GPU IDs that this Ray process
                                        can use
        -g, --num-gpus INTEGER          the number of GPUs on this node
        -H, --head                      provide this argument for the head node
        -t, --temp-dir TEXT             manually specify the root temporary dir of
                                        the Ray process, only works when --head is
                                        specified
        -m, --metrics-export-port INTEGER   
                                        the port to use to expose Ray metrics
                                        through a Prometheus endpoint.
USAGE
    exit 1
}
# if no arguments are provided, return usage function
if [ $# -eq 0 ]; then
    usage # run usage function
    exit 1
fi

OBJECT_STORE_MEMORY=
NUM_CPUS=
NUM_GPUS=
VISIBLE_DEVICES=
HEAD=false
TEMP_DIR=
METRICS_EXPORT_PORT=

while [ "$1" != "" ]; do
    case $1 in
    -o | --object-store-memory)
        shift # remove `-o` or `--object-store-memory` from `$1`
        OBJECT_STORE_MEMORY=$1
        ;;
    -c | --num-cpus)
        shift # remove `-c` or `--num-cpus` from `$1`
        NUM_CPUS=$1
        ;;
    -G | --visible-devices)
        shift # remove `-G` or `--visible-devices` from `$1`
        VISIBLE_DEVICES=$1
        ;;
    -g | --num-gpus)
        shift # remove `-g` or `--num-gpus` from `$1`
        NUM_GPUS=$1
        ;;
    -H | --head)
        HEAD=true
        ;;
    -t | --temp-dir)
        shift # remove `-t` or `--temp-dir` from `$1`
        TEMP_DIR=$1
        ;;
    -m | --metrics-export-port)
        shift # remove `-m` or `--metrics-export-port` from `$1`
        METRICS_EXPORT_PORT=$1
        ;;
    -h | --help)
        usage # run usage function
        ;;
    *)
        usage
        exit 1
        ;;
    esac
    shift # remove the current value for `$1` and use the next
done

if [[ $OBJECT_STORE_MEMORY != "" ]]; then
    OBJECT_STORE_MEMORY="--object-store-memory $OBJECT_STORE_MEMORY";
fi
if [[ $NUM_CPUS != "" ]]; then
    NUM_CPUS="--num-cpus $NUM_CPUS";
fi
if [[ $NUM_GPUS != "" ]]; then
    NUM_GPUS="--num-gpus $NUM_GPUS";
fi
if [[ $HEAD == true ]]; then
    HEAD="--head";
else
    HEAD="";
fi
if [[ $TEMP_DIR != "" ]]; then
    TEMP_DIR="--temp-dir $TEMP_DIR";
fi
if [[ $METRICS_EXPORT_PORT != "" ]]; then
    METRICS_EXPORT_PORT="--metrics-export-port $METRICS_EXPORT_PORT";
fi

# Using the provided arguments, start the Ray cluster
ulimit -n 65536;export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES; ray start $OBJECT_STORE_MEMORY $NUM_CPUS $NUM_GPUS $HEAD $TEMP_DIR $METRICS_EXPORT_PORT
