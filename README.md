# MLaaS_For_Distribution

## How to run the code

### Install the required packages using the following command:
```
conda env create -f environment.yml
```

### Activate the environment using the following command:
```
conda activate ps
```

### Change configuration in `web_server/config.py` and `mlaas/config.py` to match your database.

### At Parameter Server Node, run the following commands:

For parameter server controller:
```
cd mlaas/
python server_controller.py --network_interface <network_interface>
```

For web server:
```
cd web_server/
python web_server.py
```

### At Worker Node, run the following command:

```
cd mlaas/
python worker_controller.py --network_interface <network_interface>
```