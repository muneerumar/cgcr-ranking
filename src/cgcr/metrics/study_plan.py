"""Wilson precision planning from dev effects; no test-data access."""
import math


def wilson_interval(p,n,z=1.959963984540054):
    if not 0<=p<=1 or n<1:raise ValueError('A probability and positive sample size are required')
    denominator=1+z*z/n;centre=(p+z*z/(2*n))/denominator
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0.,centre-half),min(1.,centre+half)


def required_queries(p,n,half_width=.08):
    """Plan conservatively over the pilot Wilson interval, per cell, not simultaneous."""
    if not 0<half_width<.5:raise ValueError('Half width must lie between zero and 0.5')
    low,high=wilson_interval(p,n);worst=min(max(.5,low),high)
    for size in range(2,1_000_001):
        a,b=wilson_interval(worst,size)
        if (b-a)/2<=half_width:return dict(pilot_p=p,pilot_n=n,pilot_wilson95=[low,high],planning_p=worst,queries=size,half_width=(b-a)/2)
    raise ValueError('Precision request too large')


def cell_hours(categories,models,queries,orders,conditions,seconds_per_call):
    if any(v<0 for v in [categories,models,queries,orders,conditions,seconds_per_call]):raise ValueError('Nonnegative inputs required')
    return categories*models*queries*orders*conditions*seconds_per_call/3600
