from agora_wot.gateway import DataGateway

from agora_gw.gateway.eco import EcoGateway


def discover_seeds(eco_gw, query, host='agora', port=80, **kwargs):
    # type: (EcoGateway, str, str, int, any) -> set
    ted = eco_gw.discover(query, lazy=False)
    dgw = DataGateway(eco_gw.agora, ted, cache=None, port=port, server_name=host)
    seeds = dgw.proxy.instantiate_seeds(**kwargs)
    seed_uris = set(reduce(lambda x, y: x + y, seeds.values(), []))
    return seed_uris

