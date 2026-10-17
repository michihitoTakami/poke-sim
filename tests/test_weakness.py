from pokesim.evaluation.weakness import analyse_record, species_aliases, summarise


def aliases():
    return species_aliases({'species': [
        {'name': 'Salamence', 'baseSpecies': 'Salamence'},
        {'name': 'Salamence-Mega', 'baseSpecies': 'Salamence', 'battleOnly': 'Salamence'},
        {'name': 'Ninetales-Alola', 'baseSpecies': 'Ninetales'},
        {'name': 'Floette-Mega', 'battleOnly': 'Floette-Eternal'}]})


def test_appeared_species_and_moves_have_one_denominator_per_battle():
    rec = {'result': 'loss', 'lineup': ['Floette-Eternal'], 'log': [
        '|poke|p2|Ninetales-Alola, L50|', '|switch|p1a: Floette|Floette-Eternal, L50|173/173',
        '|switch|p2a: Mence|Salamence, L50|100/100', '|turn|1',
        '|move|p2a: Mence|Double-Edge|p1a: Floette', '|-damage|p1a: Floette|50/173',
        '|switch|p2a: Mence|Salamence-Mega, L50|100/100', '|turn|2',
        '|move|p2a: Mence|Double-Edge|p1a: Floette', '|-damage|p1a: Floette|0 fnt',
        '|faint|p1a: Floette']}
    battle = analyse_record(rec, aliases())
    unknown = analyse_record(dict(rec, result=None), aliases())
    summary = summarise([battle, unknown])
    assert battle['opponent_preview'] == ['Ninetales-Alola']
    assert summary['by_seen'] == [{'name': 'Salamence', 'battles': 1, 'wins': 0, 'losses': 1, 'win_rate': 0.}]
    assert summary['by_move'][0]['battles'] == 1
    assert summary['direct_ko_moves'] == {'Double-Edge': 1}
    assert summary['known_results'] == 1


def test_residual_ko_does_not_become_the_last_attacks_ko():
    rec = {'result': 'loss', 'lineup': ['Scolipede', 'Gholdengo'], 'log': [
        '|switch|p1a: Gholdengo|Gholdengo, L50|175/175',
        '|switch|p2a: Dragon|Archaludon, L50|100/100', '|turn|1',
        '|-weather|RainDance', '|move|p2a: Dragon|Electro Shot||[still]',
        '|-damage|p1a: Gholdengo|1/175', '|-damage|p1a: Gholdengo|0 fnt|[from] Curse',
        '|faint|p1a: Gholdengo']}
    battle = analyse_record(rec, {})
    assert battle['own_faints'][0]['source'] == 'residual_or_unknown'
    assert {m['kind'] for m in battle['markers']} == {'last_scolipede', 'rain_electro_shot'}
    assert summarise([battle])['direct_ko_moves'] == {}


def test_inconsistent_lineup_suppresses_remaining_roster_inferences():
    rec = {'result': 'loss', 'lineup': ['Scolipede', 'Gholdengo'], 'log': [
        '|switch|p1a: Floette|Floette-Mega, L50|173/173',
        '|switch|p1a: Gholdengo|Gholdengo, L50|175/175', '|faint|p1a: Gholdengo']}
    battle = analyse_record(rec, aliases())
    assert battle['lineup_mismatch']
    assert 'last_scolipede' not in {m['kind'] for m in battle['markers']}
