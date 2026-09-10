def choices(domain):
    suffixes = ['']
    for switch in domain.switches:
        suffixes = [s + extra for s in suffixes for extra in ('', '+' + switch.parameter.replace('_', '-'))]
    return tuple(model + '/' + design + suffix
                 for model, designs in domain.designs.items() for design in designs for suffix in suffixes)


def switch_parameters(domain):
    return [switch.parameter for switch in domain.switches]


def structure_metadata(domain, version):
    return dict(version=version, model='categorical',
                shared_binary=switch_parameters(domain) + [m.parameter for m in domain.registry.modifiers],
                conditional_layouts=sorted(domain.registry.layouts),
                model_parameters='Shared across compatible graph variants within each model.')


def structure(domain, config, categorical):
    model, design = config['model'], config['design']
    parts = design.split('+')
    active = set(parts[1:])
    values = dict(model=model)
    domains = dict(model=categorical(domain.models))
    for switch in domain.switches:
        values[switch.parameter] = config.get(switch.key, switch.inactive) == switch.active
        domains[switch.parameter] = categorical([False, True])
    parameter = domain.layout_parameter(model)
    if parameter is not None:
        values[parameter] = parts[0]
        domains[parameter] = categorical(list(domain.registry.layouts[parameter]))
    for modifier in domain.modifiers(model):
        values[modifier.parameter] = modifier.name in active
        domains[modifier.parameter] = categorical([False, True])
    return values, domains


def suggest_structure(domain, trial, models=None):
    model = trial.suggest_categorical('model', list(models or domain.models))
    context = {}
    for switch in domain.switches:
        chosen = trial.suggest_categorical(switch.parameter, [False, True])
        context[switch.key] = switch.active if chosen else switch.inactive
    options = domain.bases(model)
    parameter = domain.layout_parameter(model)
    design = trial.suggest_categorical(parameter, list(domain.registry.layouts[parameter])) if parameter is not None else options[0]
    for modifier in domain.modifiers(model):
        if trial.suggest_categorical(modifier.parameter, [False, True]):
            design += '+' + modifier.name
    return model, design, context
