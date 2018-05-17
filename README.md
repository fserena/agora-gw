# Agora Gateway


A semantic gateway for the Web of Things.

## Getting started

### Example 1: Creating a Gateway object

This example assumes there is no external repository nor Agora to connect to.

```python
from agora_gw import Gateway

gw = Gateway()
```

### Example 2: Creating a Gateway object connected to a GraphDB instance

This example assumes that a GraphDB instance is deployed to localhost:7200.

```python
from agora_gw import Gateway

config = {
    'description': {
        'query_url': 'http://localhost:7200/repositories/tds',
        'update_url': 'http://localhost:7200/repositories/tds/statements'
    }        
}

gw = Gateway(**config)
```
### Example 3: Loading a semantic extension about movies

```python
from StringIO import StringIO

from agora_gw import Gateway
from rdflib import Graph

extension_ttl = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dbo: <http://dbpedia.org/ontology/> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .

dbo:starring rdf:type owl:ObjectProperty .

dbo:director rdf:type owl:ObjectProperty ;
    rdfs:domain dbo:Film ;
    rdfs:range foaf:Person .

dbo:Film rdf:type owl:Class ;
     rdfs:subClassOf [ rdf:type owl:Restriction ;
                       owl:onProperty dbo:starring ;
                       owl:someValuesFrom foaf:Person
                     ] ,
                     _:named .

foaf:Person rdf:type owl:Class ;
    rdfs:subClassOf _:named  .

_:named a owl:Restriction ;
    owl:onProperty foaf:name ;
    owl:onDataRange xsd:string .

foaf:name a owl:DatatypeProperty .

rdfs:label a owl:DatatypeProperty ;
    rdfs:range xsd:string .

dbo:spouse a owl:ObjectProperty ;
    rdfs:domain foaf:Person ;
    rdfs:range foaf:Person .    
"""

gw = Gateway()
extension_g = Graph()
extension_g.parse(StringIO(extension_ttl), format='turtle')
gw.add_extension('movies', extension_g)

print gw.extensions # ['movies']
print gw.agora.fountain.types   # ['dbo:Film', 'foaf:Person'] 
print gw.agora.fountain.properties  # ['rdfs:label', 'dbo:spouse', 'dbo:starring', 'foaf:name', 'dbo:director'] 
```


agora-gw is distributed under the Apache License, version 2.0.
