from dataclasses import dataclass

TARGET = "retained"
DATES = ("pol_policy_effective_date", "pol_no_longer_in_force_date", "pol_expiration_date")
EMPTY = frozenset(("", "null", "none", "nan", "-1", "0"))
NUMBERS = [
    "age_days",
    "active",
    "ended_days_ago",
    "cancelled",
    "policies",
    "active_policies",
    "past_cancellations",
    "tenure_days",
]
ALIASES = {"branch_key": "db_branch_key", "servicer_key": "ds_servicer_key"}
RENEWAL = "expiring_policy_key"


def present(value):
    return value.strip().casefold() not in EMPTY


@dataclass(frozen=True)
class Entity:
    field: str
    kind: str
    categorical: bool = False
    client_link: bool = False
    hierarchy_width: int = 0
    namespace: str | None = None

    def keys(self, columns):
        values = [value.strip() for value in columns[self.field]]
        if self.namespace is None:
            return values
        return [
            prefix.strip() + ":" + value.casefold() if present(prefix) and present(value) else ""
            for prefix, value in zip(columns[self.namespace], values)
        ]

    def members(self, keys):
        members = set(filter(present, keys))
        parents = {
            code[:width]
            for code in members
            if code.isdigit() and 2 <= len(code) <= self.hierarchy_width
            for width in range(2, len(code))
        }
        return sorted(members | parents)


ENTITIES = (
    Entity("producer_key", "producer"),
    Entity("placement_specialist_key", "placement_specialist"),
    Entity("account_executive_key", "account_executive"),
    Entity("branch_key", "branch"),
    Entity("billing_company_key", "billing_company"),
    Entity("issuing_company_key", "issuing_company"),
    Entity("servicer_key", "servicer"),
    Entity("policy_type_key", "policy_type"),
    Entity("client_state_code", "client_state", categorical=True, client_link=True),
    Entity("client_city", "client_city", categorical=True, client_link=True, namespace="client_state_code"),
    Entity("issuing_state_code", "issuing_state", categorical=True),
    Entity("naics_code", "naics", categorical=True, client_link=True, hierarchy_width=6),
    Entity("sic_code", "sic", categorical=True, client_link=True, hierarchy_width=4),
    Entity("integrated_platform", "platform", categorical=True),
    Entity("pol_income_type", "income_type", categorical=True),
    Entity("line_of_business_code", "line_of_business", categorical=True),
    Entity("policy_type_group_code", "policy_group", categorical=True),
)
BY_FIELD = {entity.field: entity for entity in ENTITIES}
KNOWN_FIELDS = tuple(BY_FIELD) + (RENEWAL,)
RELATIONSHIPS = (
    ("policy_type_key", "policy_type_group_code", "type_group"),
    ("policy_type_group_code", "line_of_business_code", "group_business"),
    ("client_city", "client_state_code", "city_state"),
)


def selected_entities(fields, available):
    unsupported = set(fields) - set(KNOWN_FIELDS)
    missing = set(fields) - set(available)
    if unsupported or missing:
        raise ValueError("unsupported or missing graph fields: " + ", ".join(sorted(unsupported | missing)))
    if len(set(fields)) != len(fields):
        raise ValueError("graph fields must be unique")
    selected = [BY_FIELD[field] for field in fields if field in BY_FIELD]
    for entity in selected:
        if entity.namespace is not None and entity.namespace not in fields:
            raise ValueError(entity.field + " requires " + entity.namespace + " in known fields")
    return selected
