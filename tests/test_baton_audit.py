from pokesim.evaluation.baton import audit


def test_audit_keeps_passed_stages_and_separates_attempts_losses_and_receivers():
    log = ['|switch|p1a: Bug|Scolipede, L50|165/165', '|turn|1',
           '|-boost|p1a: Bug|def|2', '|-boost|p1a: Bug|spe|1',
           '|move|p1a: Bug|Baton Pass|p1a: Bug',
           '|switch|p1a: Moon|Umbreon, L50|190/190', '|turn|2',
           '|-boost|p1a: Moon|spa|1', '|-boost|p1a: Moon|spd|1',
           '|switch|p1a: Gold|Gholdengo, L50|175/175', '|turn|3',
           '|faint|p1a: Gold', '|switch|p1a: Bug|Scolipede, L50|40/165', '|turn|4',
           '|-boost|p1a: Bug|spe|1', '|faint|p1a: Bug']
    rec = dict(log=log, lineup=['Scolipede', 'Umbreon', 'Gholdengo'], result='loss')
    out = audit([('battle.json', rec)])
    assert out['totals']['attempted_passes'] == out['totals']['completed_passes'] == 1
    assert out['recipients'] == {'umbreon': 1}
    assert out['totals']['boosts_lost:switch'] == out['totals']['boosts_lost:faint'] == 1
    assert out['examples']['boosts_lost:switch'][0]['boosts'] == {'def': 2, 'spe': 1, 'spa': 1, 'spd': 1}


def test_a_failed_dry_pass_is_not_a_completed_handoff():
    rec = dict(result='loss', lineup=['Scolipede'], log=[
        '|switch|p1a: Bug|Scolipede, L50|165/165', '|turn|1',
        '|move|p1a: Bug|Baton Pass|p1a: Bug', '|-fail|p1a: Bug', '|faint|p1a: Bug'])
    out = audit([('battle.json', rec)])
    assert out['totals']['attempted_passes'] == out['totals']['pass_without_stages_or_substitute'] == 1
    assert out['totals'].get('completed_passes', 0) == 0


def test_forced_removal_is_separate_from_a_voluntary_switch():
    rec = dict(result='loss', lineup=['Scolipede', 'Umbreon'], log=[
        '|switch|p1a: Bug|Scolipede, L50|165/165', '|turn|1', '|-boost|p1a: Bug|def|2',
        '|drag|p1a: Moon|Umbreon, L50|190/190'])
    out = audit([('battle.json', rec)])
    assert out['totals']['boosts_lost:drag'] == 1
    assert out['totals'].get('boosts_lost:switch', 0) == 0
