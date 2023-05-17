if command -v tmux &> /dev/null; then
    AIM_SVC=$(kubectl get svc | grep aim-service | awk {'print $1'})
    # Create tmux session for the Aim server
    if ! tmux has-session -t aim-server &> /dev/null; then
        tmux new -s aim-server -d
    fi
    # Kill any previous port-forwarding, then port-forward the service
    tmux send-keys -t aim-server.0 C-c
    tmux send -t aim-server.0 \
    "kubectl port-forward svc/$AIM_SVC 43800:43800" Enter

    HEAD_SVC=$(kubectl get svc --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
    # Create tmux session for the Ray dashboard
    if ! tmux has-session -t ray-dashboard &> /dev/null; then
        tmux new -s ray-dashboard -d
    fi
    # Kill any previous port-forwarding, then port-forward the service
    tmux send-keys -t ray-dashboard.0 C-c
    tmux send -t ray-dashboard.0 \
    "kubectl port-forward svc/$HEAD_SVC 8265:8265" Enter

    # Create tmux session for the Ray client server
    if ! tmux has-session -t ray-client-server &> /dev/null; then
        tmux new -s ray-client-server -d
    fi
    # Kill any previous port-forwarding, then port-forward the service
    tmux send-keys -t ray-client-server.0 C-c
    tmux send -t ray-client-server.0 \
    "kubectl port-forward svc/$HEAD_SVC 10001:10001" Enter

    HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
    # Create tmux session for the Prometheus dashboard
    if ! tmux has-session -t prometheus-dashboard &> /dev/null; then
        tmux new -s prometheus-dashboard -d
    fi
    # Kill any previous port-forwarding, then port-forward the service
    tmux send-keys -t prometheus-dashboard.0 C-c
    tmux send -t prometheus-dashboard.0 \
    "kubectl port-forward pods/$HEAD_POD 3000:3000" Enter

    # Create tmux session for the Grafana dashboard
    if ! tmux has-session -t grafana-dashboard &> /dev/null; then
        tmux new -s grafana-dashboard -d
    fi
    # Kill any previous port-forwarding, then port-forward the service
    tmux send-keys -t grafana-dashboard.0 C-c
    tmux send -t grafana-dashboard.0 \
    "kubectl port-forward pods/$HEAD_POD 9090:9090" Enter
fi
