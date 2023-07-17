# Assumption: strictly 1 raycluster is currently deployed on AIP
# HEAD_POD=$(kubectl get pods | grep raycluster | grep head | awk '{ print $1 }')
HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)

# Create a tmux session for prometheus on the head node
kubectl exec $HEAD_POD -- bash -c "if ! tmux has-session -t prometheus &> /dev/null; \
    then tmux new -s prometheus -d; fi"
# Start prometheus on the head node from within the prometheus tmux session
kubectl exec $HEAD_POD -- bash -c "tmux send -t prometheus.0 \
    'cd /home/ray/prometheus-2.43.0.linux-amd64 && \
    ./prometheus --config.file=/home/ray/prometheus.yml' Enter"

# Create a tmux session for grafana on the head node
kubectl exec $HEAD_POD -- bash -c "if ! tmux has-session -t grafana &> /dev/null; \
    then tmux new -s grafana -d; fi"
# Start grafana-server on the head node from within the grafana tmux session
kubectl exec $HEAD_POD -- bash -c "tmux send -t grafana.0 \
    'cd /home/ray/grafana-9.4.7 && \
    ./bin/grafana-server --config /home/ray/grafana.ini web' Enter"
