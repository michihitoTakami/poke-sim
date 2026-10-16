"""Exercise the actual actor-observation policy and simulator, including safe handoffs and lethal replies."""
import copy
import json
import subprocess

import pytest
from test_expert_continuation import CHOMP, FLOETTE, GHOLDENGO, PASSER, SPEC, WEAK, side

from pokesim.config import ROOT, SHOWDOWN_DIR

UMBREON = dict(WEAK, species='Umbreon', ability='Inner Focus', nature='Bold',
               moves=['Snarl', 'Baton Pass', 'Calm Mind', 'Moonlight'])


def public_choice(sample, *, passing=False, old=False, history=None):
    js = r"""
const {createCodec}=require('./services/sim-tools/state_codec');
const {PublicObservation}=require('./services/sim-tools/public_observation');
const {choosePublic,chooseBatonPublic}=require('./services/sim-tools/public_policy');
const {extractChannelMessages}=require(process.argv[1]+'/dist/sim/battle');
const args=JSON.parse(process.argv[2]), b=createCodec(process.argv[1]).setUp(args.sample,'gen9championsbssregmc',1);
const log=[];
for(const side of b.sides) {
 const m=side.active[0];
 log.push(`|switch|${side.id}a: ${m.name}|${m.species.name}, L50|${m.hp}/${m.maxhp}`);
}
log.push('|turn|1');
for(const side of b.sides) {
 const m=side.active[0], who=side.id+'a: '+m.name;
 for(const [s,n] of Object.entries(m.boosts)) if(n) log.push(`|-setboost|${who}|${s}|${n}`);
 for(const v of Object.keys(m.volatiles)) log.push(`|-start|${who}|${v==='substitute'?'Substitute':v}`);
 if(m.lastMove) log.push(`|move|${who}|${m.lastMove.name}|${who}`);
 // All of these moves are deliberately revealed in these tactical regression fixtures.
 if(side.id==='p2') for(const s of m.moveSlots) log.push(`|move|${who}|${s.move}|p1a: ${b.p1.active[0].name}`);
}
if(args.sample.field.trickroom) log.push('|-fieldstart|move: Trick Room');
log.push('|turn|2',...(args.history||[]));
const obs=new PublicObservation(log,b.dex,extractChannelMessages), view=obs.view(b.p1);
if(args.passing) {view.phase='switch';view.request.forceSwitch=[true];
 view.public.mons.p1[view.public.active.p1].lastMove='batonpass';}
const policy=args.old?choosePublic:chooseBatonPublic;
const before=JSON.stringify(view), result=policy(view,'expert',b.dex);
console.log(JSON.stringify({label:result[1],unchanged:before===JSON.stringify(view)}));b.destroy();
"""
    request = dict(sample=sample, passing=passing, old=old, history=history)
    out = subprocess.run(['node', '-e', js, str(SHOWDOWN_DIR), json.dumps(request)], cwd=ROOT,
                         capture_output=True, text=True, timeout=20, check=True)
    result = json.loads(out.stdout)
    assert result['unchanged']
    return result['label']


def position(ours, theirs):
    return dict(p1=ours, p2=theirs, field={})


def test_passer_hands_off_before_an_extra_setup_turn_would_be_lethal():
    sample = position(side([PASSER, GHOLDENGO], [{'hp': .3, 'boosts': {'def': 2, 'spe': 2}}, {'hp': 1}]),
                      side([CHOMP]))
    assert public_choice(sample) == 'move:batonpass'


def test_umbreon_builds_a_payload_then_relays_instead_of_spamming_snarl():
    sample = position(side([UMBREON, GHOLDENGO]), side([WEAK]))
    assert public_choice(sample) == 'move:calmmind'
    sample['p1']['mons'][0]['boosts'] = {'spa': 2, 'spd': 2, 'def': 4, 'spe': 2}
    assert public_choice(sample) == 'move:batonpass'


def test_no_dry_pass_when_there_is_time_to_build():
    assert public_choice(position(side([PASSER, FLOETTE]), side([WEAK]))) != 'move:batonpass'


def test_no_second_consecutive_protect():
    ours = side([PASSER, FLOETTE], [{'hp': .8, 'boosts': {'spe': 1}, 'lastMove': 'protect', 'stall': 1}, {'hp': 1}])
    assert public_choice(position(ours, side([dict(CHOMP, moves=['Flamethrower'])]))) != 'move:protect'


def test_a_safe_umbreon_relay_is_allowed_but_not_a_pass_back_to_scolipede():
    sample = position(side([PASSER, UMBREON, GHOLDENGO],
                           [{'hp': 1, 'boosts': {'def': 4, 'spe': 2}}, {'hp': 1}, {'hp': 1}]), side([WEAK]))
    assert public_choice(sample, passing=True) == 'switch:umbreon'
    sample['p1'] = side([UMBREON, PASSER, GHOLDENGO],
                        [{'hp': 1, 'boosts': {'def': 4, 'spe': 2, 'spa': 2, 'spd': 2}}, {'hp': 1}, {'hp': 1}])
    assert public_choice(sample, passing=True) == 'switch:gholdengo'


@pytest.mark.parametrize('boosts,target', [({'atk': 4}, 'switch:gyarados'), ({'spa': 4}, 'switch:gholdengo')])
def test_receivers_are_evaluated_using_the_stats_actually_being_passed(boosts, target):
    physical = dict(CHOMP, species='Gyarados', ability='Intimidate', moves=['Waterfall'])
    special = dict(GHOLDENGO, moves=['Flash Cannon'])
    sample = position(side([PASSER, physical, special], [{'hp': 1, 'boosts': boosts}, {'hp': 1}, {'hp': 1}]),
                      side([dict(WEAK, species='Mew', ability='Synchronize',
                                 moves=['Splash', 'Soft-Boiled', 'Protect', 'Heal Bell'])]))
    assert public_choice(sample, passing=True) == target


def test_lethal_priority_cannot_be_outrun_by_speed_boosts():
    sample = position(side([PASSER, GHOLDENGO], [{'hp': .05, 'boosts': {'spe': 6}}, {'hp': 1}]),
                      side([dict(CHOMP, moves=['Extreme Speed'])]))
    assert public_choice(sample) == 'switch:gholdengo'


CHANDELURE = dict(GHOLDENGO, species='Chandelure', item='', ability='Flash Fire', nature='Modest',
                  moves=['Flamethrower'], evs=SPEC)


@pytest.mark.parametrize('move,species,foe', [
    ('shadowball', GHOLDENGO, CHANDELURE), ('moonblast', FLOETTE, CHOMP),
])
def test_a_lethal_attack_precedes_recovery_or_setup(move, species, foe):
    sample = position(side([species], [{'hp': .35, 'boosts': {'spa': 2, 'spe': 2}}]),
                      side([foe], [{'hp': .2}]))
    assert public_choice(sample) == 'move:' + move


def test_hidden_set_changes_do_not_change_the_new_public_policy():
    sample = position(side([PASSER, GHOLDENGO]), side([CHOMP]))
    # Opponent private nature/item/spread differ; only its revealed moves are supplied to the observer.
    changed = copy.deepcopy(sample)
    changed['p2']['team'][0].update(item='Choice Scarf', nature='Brave')
    assert public_choice(sample) == public_choice(changed)


def test_a_sampled_revival_blessing_continues_through_the_actual_revive_request():
    from test_worker_transitions import worker

    pawmot = dict(CHOMP, species='Pawmot', ability='Natural Cure', moves=['Revival Blessing'])
    sample = position(side([PASSER, GHOLDENGO]), side([pawmot, CHOMP], [{'hp': 1}, {'hp': 0}]))
    code = r"""
const {chooseBatonPublic}=require(process.argv[1].replace('lookahead.js','public_policy.js'));
const {PublicObservation,legalChoices}=require(process.argv[1].replace('lookahead.js','public_observation.js'));
const {extractChannelMessages}=require(process.argv[2]+'/dist/sim/battle');
const b=setUp(sample,format,1);
// Both requests come from the real engine, rather than a manually invented revive flag.
b.makeChoices('move 1','move 1');
const history=['|switch|p2a: Pawmot|Pawmot, L50|100/100','|turn|1'];
const observer=new PublicObservation(history,b.dex,extractChannelMessages);
const view=observer.view(b.p2), opts=legalChoices(view), chosen=chooseBatonPublic(view,'expert',b.dex);
const engineOptions=choices(b,b.p2);
b.makeChoices('pass',chosen[0]);
console.log(JSON.stringify({opts,engineOptions,chosen,revived:b.p2.pokemon[1].hp>0}));b.destroy();
"""
    out = worker(code, sample)
    assert out['opts'] == out['engineOptions'] == [['switch 2', 'switch:garchomp']]
    assert out['chosen'] == out['opts'][0] and out['revived']
