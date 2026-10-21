"""Two independent numerical requirements; all cohorts must pass."""
import math


def decision(coherent_nll, shuffled_nll, frequency_nll, settings):
    if not all(math.isfinite(v) for v in (coherent_nll, shuffled_nll, frequency_nll)):
        raise ValueError('Non-finite readiness score')
    f, o = frequency_nll-coherent_nll, shuffled_nll-coherent_nll
    return {'frequency_advantage_nats': f, 'ordering_advantage_nats': o,
            'frequency_passed': f >= settings['minimum_frequency_advantage_nats'],
            'ordering_passed': o >= settings['minimum_ordering_advantage_nats'],
            'passed': f >= settings['minimum_frequency_advantage_nats'] and o >= settings['minimum_ordering_advantage_nats']}
