ARG RAY_VERSION=2.4.0

# Deployment Stage
FROM butler.mitre.org/fireline/simharness2:simple-${RAY_VERSION}

# Copy the needed code
WORKDIR $HOME
# Install Grafana v9.4.7 from binary .tar.gz file
RUN wget --no-check-certificate \
    https://dl.grafana.com/enterprise/release/grafana-enterprise-9.4.7.linux-amd64.tar.gz \
    && tar -zxvf grafana-enterprise-9.4.7.linux-amd64.tar.gz
# Install Prometheus v2.43.0 from binary .tar.gz file
RUN wget --no-check-certificate \
    https://github.com/prometheus/prometheus/releases/download/v2.43.0/prometheus-2.43.0.linux-amd64.tar.gz \
    && tar -zxvf prometheus-2.43.0.linux-amd64.tar.gz
# Remove the installers that were downloaded above
RUN rm \
    grafana-enterprise-9.4.7.linux-amd64.tar.gz \
    prometheus-2.43.0.linux-amd64.tar.gz

# Copy the needed code
WORKDIR /code/
COPY docker/grafana/grafana.ini docker/prometheus/prometheus.yml $HOME/

# Install tmux to allow for launching grafana and prometheus in the background
RUN sudo apt-get install tmux
# WORKDIR /code/grafana-9.4.7
# RUN ./bin/grafana-server --config ../grafana.ini web
