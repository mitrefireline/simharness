# Assumption: strictly 1 raycluster is currently deployed on AIP
# HEAD_POD=$(kubectl get pods | grep raycluster | grep head | awk '{ print $1 }')
HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)

# Interrupt prometheus from within the prometheus tmux session
kubectl exec $HEAD_POD -- bash -c "tmux send-keys -t prometheus.0 C-c"

# Interrupt grafana from within the grafana tmux session
kubectl exec $HEAD_POD -- bash -c "tmux send-keys -t grafana.0 C-c"