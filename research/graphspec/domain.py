class Switch:

    __slots__ = ('parameter', 'key', 'active', 'inactive')

    def __init__(self, parameter, key, active, inactive):
        self.parameter, self.key = parameter, key
        self.active, self.inactive = active, inactive

    def declaration(self):
        return dict(parameter=self.parameter, key=self.key, active=self.active, inactive=self.inactive)


class Domain:

    def __init__(self, registry, graphs, blocks=None, switches=(), views=None):
        self.registry, self.graphs = registry, dict(graphs)
        self.blocks = dict(blocks or {})
        self.switches = tuple(switches)
        self.views = dict(views or {})
        for model, graph in self.graphs.items():
            if graph not in registry.layouts and graph not in registry.recipes:
                raise ValueError('model %s names an undeclared graph %s' % (model, graph))
        self.models = tuple(self.graphs)
        self.designs = {model: registry.designs_for(self.bases(model)) for model in self.graphs}

    def bases(self, model):
        graph = self.graphs[model]
        return self.registry.layouts.get(graph, (graph,))

    def layout_parameter(self, model):
        graph = self.graphs[model]
        return graph if graph in self.registry.layouts else None

    def modifiers(self, model):
        return self.registry.modifiers_for(self.bases(model))

    def view(self, model):
        return self.registry.recipes[self.bases(model)[0]].view

    def arrays(self, model):
        return tuple(self.views.get(self.view(model), ()))

    def declaration(self):
        return dict(spec=self.registry.declaration(), graphs=dict(self.graphs),
                    blocks={k: sorted(v) for k, v in self.blocks.items()},
                    switches=[s.declaration() for s in self.switches],
                    views={k: list(v) for k, v in self.views.items()})
