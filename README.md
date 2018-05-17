# Agora Gateway


A semantic gateway for the Web of Things.

## Getting started

### Example 1: Creating a Gateway object

This example assumes there is no external service to connect to.

```python
from agora_gw import Gateway

gw = Gateway()
print gw.extensions
```

### Example 2: Creating a Gateway object connected to a GraphDB instance

This example assumes that a GraphDB instance is deployed to localhost:7200.

```python
from agora_gw import Gateway

config = {
    'sparql_host': 'http://localhost:7200/repositories/tds',
    'update_host': 'http://localhost:7200/repositories/tds/statements'        
}

gw = Gateway(**config)
print gw.extensions
```


agora-gw is distributed under the Apache License, version 2.0.
