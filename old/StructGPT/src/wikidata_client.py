import requests


class WikidataEntityNotFoundException(Exception):
    pass


class WikidataClient():

    def get_entity_id(self, entity):
        """Get the entity id from Wikidata by performing a full text search and returning the first result."""
        base_url = 'https://wikidata.org/w/api.php'
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': entity
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        try:
            return data['query']['search'][0]['title']
        except IndexError:
            return None

    def get_one_hop_relations(
        self,
        tpe_id,
        max_properties_and_values=None
    ):
        """
            Get the one-hop relations of the topic entity from Wikidata.
            The relations are in the format of (relation_id, relation_name).
        """

        sparql = """
            SELECT ?wdLabel ?ps_Label {
            VALUES (?entity) {(wd:<tpe_id>)}
            
            ?entity ?p ?statement .
            ?statement ?ps ?ps_ .
            
            ?wd wikibase:claim ?p.
            ?wd wikibase:statementProperty ?ps.
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
            } ORDER BY ?wd ?statement ?ps_
        """.replace('<tpe_id>', tpe_id)

        results = self.execute_sparql(sparql)
        one_hop_relations = {}
        for result in results:
            relation_name = result['wdLabel']['value']
            relation_value = result['ps_Label']['value']

            one_hop_relations[relation_name] = one_hop_relations.get(
                relation_name, []) + [relation_value]
            if max_properties_and_values and len(one_hop_relations) >= max_properties_and_values:
                break

        return one_hop_relations

    def filter_ids(self, one_hop_relations):
        """Filter the one-hop relations to remove all the ids."""
        filtered_one_hop_relations = {}
        for relation_name, relation_values in one_hop_relations.items():
            if not "id" in relation_name.lower().split():
                filtered_one_hop_relations[relation_name] = relation_values
        return filtered_one_hop_relations

    def get_freebase_id(self, entity_id):
        """Get the freebase id from Wikidata."""
        sparql = """
            SELECT ?freebase_id WHERE {
            wd:<entity_id> wdt:P646 ?freebase_id.
            }
        """.replace('<entity_id>', entity_id)

        results = self.execute_sparql(sparql)
        try:
            return results[0]['freebase_id']['value']
        except IndexError:
            return None

    def execute_sparql(self, sparql):
        """Execute the sparql query."""
        sparql = sparql.replace('\n', ' ').replace(
            '\t', ' ').replace('\r', ' ').strip()
        url = "https://query.wikidata.org/sparql"
        r = requests.get(url, params={'format': 'json', 'query': sparql})
        data = r.json()
        return data['results']['bindings']