''' 
Present an interactive clinical trial dimensioning interface

'''
from binoculars import binomial_confidence
from scipy import stats
import math
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, BoxAnnotation, HoverTool, Button, Spacer, Div, Whisker, Range1d
from bokeh.plotting import figure

### Configuration

PLOT_RANGE_DELTA = 5

GROUPS = ['Test group', 'Control group']

# Groups
CONTROL_MIN   = 50
CONTROL_MAX   = 25000
CONTROL_START = 1000
CONTROL_STEP  = 10

TEST_MIN   = 50
TEST_MAX   = 25000
TEST_START = 1000
TEST_STEP  = 10

# Proportions of positive events
EVENTS_CONTROL_MIN   = 0.25 # this can never be zero for the control group, otherwise the would be nothing to investigate
EVENTS_CONTROL_MAX   = 100
EVENTS_CONTROL_START = 2.5
EVENTS_CONTROL_STEP  = 0.25

EVENTS_TEST_MIN   = 0
EVENTS_TEST_MAX   = 100
EVENTS_TEST_START = 1.25
EVENTS_TEST_STEP  = 0.25

# Confidence level
CI_MIN   = 90
CI_MAX   = 99
CI_START = 95
CI_STEP  = 0.5

# The Walter method is used when this variable is true. When it is false we use the simpler Katz method, but it crashes when the proportion for the test group is zero
# Katz et al, 1978 and Walter, 1975 are summarized here:
# https://www.jstor.org/stable/2531848
WALTER_CI = True

# labels and strings
PAGE_TITLE  ='Clinical trial simulator'

CONTROL_LABEL = 'Control group size'
TEST_LABEL    = 'Test group size'

EVENTS_CONTROL_LABEL = 'Detected proportion for the control group (%)'
EVENTS_TEST_LABEL    = 'Detected proportion for the test group (%)'
CI_LABEL             = 'Target confidence level (%)'

LMARGIN_WIDTH=20
MMARGIN_WIDTH=50

TEXT_WIDTH = 100
TEXT_INTRO   = 'Use the mouse for initial selection and cursors for fine tuning:'
TEXT_RESULTS = '<b>Inference from experimental results</b>'

### End of configuration

### Functions

def mk_risk_str ( title, risk, risk_l, risk_r ):

    str_risk = title + str(risk) + ' (' +  str(risk_l) + '-' + str(risk_r) + ')'

    return str_risk

def get_phi ( p0, p1, n0, n1, walter = False ):

    if not walter:
        return p1 / p0
    else:
        x0 = p0 * n0
        x1 = p1 * n1
        return math.exp( math.log( (x1 + 0.5) / (n1 + 0.5 ) ) - math.log( (x0 + 0.5) / (n0 + 0.5 ) ) )

def get_par ( p0, p1, n0, n1, walter = False ):

    if not walter:
        par = math.sqrt( (1 - p0)/(n0 * p0) + (1 - p1)/(n1 * p1) )
        return par
    else:
        x0 = p0 * n0
        x1 = p1 * n1
        par = math.sqrt( 1/(x1 + 0.5) - 1/(n1 + 0.5) + 1/(x0 + 0.5) - 1/(n0 + 0.5) )
        return par

# find interval overlaps, each interval is a list with 2 elements
def get_overlap ( i1, i2 ):

    a, b = max(i1[0], i2[0]), min(i1[-1], i2[-1])

    if a > b:
        return []
    else:
        return [a, b]

# callback function for updating the data
def update_data(attrname, old, new):

    confidence_level = ci.value / 100
    z_value = stats.norm.isf( (1 - confidence_level) / 2 )

    # control group inference
    control_risk     = round(events_control.value,2)
    control_risk_ci  = np.array( binomial_confidence(control_risk / 100, control.value, z=z_value) ) * 100
    control_risk_l   = round(control_risk_ci[0], 2)
    control_risk_r   = round(control_risk_ci[1], 2)
    control_risk_err = control_risk_ci[1] - control_risk_ci[0]

    str_control_risk = mk_risk_str ('Risk on control group (%) : ', control_risk, control_risk_l, control_risk_r)

    # test group inference
    test_risk     = round(events_test.value,2)
    test_risk_ci  = np.array( binomial_confidence(test_risk / 100, test.value, z=z_value) ) * 100
    test_risk_l   = round(test_risk_ci[0], 2)
    test_risk_r   = round(test_risk_ci[1], 2)
    test_risk_err = test_risk_ci[1] - test_risk_ci[0]

    spacing = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp'
    str_test_risk = mk_risk_str ('Risk on test group (%) ' + spacing + ': ', test_risk, test_risk_l, test_risk_r)

    tmp_interval = get_overlap ( test_risk_ci, control_risk_ci )

    if len(tmp_interval) > 0:
        overlap_length   = tmp_interval[1] - tmp_interval[0]
        overlap_interval = [ round(tmp_interval[0],2), round(tmp_interval[1],2) ]
    else:
        overlap_length = 0
        overlap_interval = []

    str_overlap_interval = 'Overlap: ' + str(round(overlap_length,2)) + ' ' +  str(overlap_interval)
    str_overlap_pct_test = 'Overlap % for test group: ' + str( round( (overlap_length / test_risk_err)*100, 2) )

    # risk ratio
    # the form is phi * math.exp( +-z_value * parameter ) which is common to Katz and Walter methods

    phi = get_phi (control_risk/100, test_risk/100, control.value, test.value, WALTER_CI)
    par = get_par (control_risk/100, test_risk/100, control.value, test.value, WALTER_CI)

    risk_ratio   = round(test_risk / control_risk,2)
    risk_ratio_l = round(phi * math.exp( -z_value * par ),2)
    risk_ratio_r = round(phi * math.exp( +z_value * par ),2)

    str_risk_ratio = mk_risk_str ('Relative risk: ', risk_ratio, risk_ratio_l, risk_ratio_r)

    # adverse effects threshold
    # to find at least one case at the current confidence level the probability must be this or higher
    adv_effects_threshold = (1 - ( 1 - confidence_level )**( 1 / test.value ) ) * 100
    str_adv_effects = 'Adverse effects detectability threshold (%): ' + str( round (adv_effects_threshold, 2) )

    text_risk.text        = '<br/>' + str_test_risk + '<br/>' + str_control_risk + '<br/><br/>' + str_overlap_interval + '<br/>' + str_overlap_pct_test
    text_risk_ratio.text  = str_risk_ratio
    text_adv_effects.text = str_adv_effects

    # produce warnings in case they are necessary
    if 1 >= risk_ratio_l and 1 <= risk_ratio_r:
        warning1 = 'The confidence interval for Relative Risk contains 1.'
    else:
        warning1 = ''

    if adv_effects_threshold > control_risk:
        warning2 = 'The adverse effects detectability threshold for the test group is above the risk level for the control group.'
    else:
        warning2 = ''

    if warning1 or warning2:
        str_warnings = '<b>Warnings:</b>\n\n' + warning1 + '&nbsp;' + warning2
    else:
        str_warnings = ''

    text_warnings.text = str_warnings

    values = [ test_risk, control_risk     ]
    upper  = [ test_risk_r, control_risk_r ]
    lower  = [ test_risk_l, control_risk_l ]

    source.data = dict(groups=GROUPS, values=values, upper=upper, lower=lower)

    p.y_range.end = math.ceil( max (upper[0], upper[1]) ) + PLOT_RANGE_DELTA

def reset_data():

    control.value = CONTROL_START
    test.value    = TEST_START

    events_control.value = EVENTS_CONTROL_START
    events_test.value    = EVENTS_TEST_START

    ci.value = CI_START

    # we seem to need to pass something here because the slider callback needs to have a declaration of 3 parameters
    update_data('xxxx',0,0)

### Main

# Group size sliders
control = Slider(title=CONTROL_LABEL, value=CONTROL_START, start=CONTROL_MIN, end=CONTROL_MAX, step=CONTROL_STEP)
test    = Slider(title=TEST_LABEL, value=TEST_START, start=TEST_MIN, end=TEST_MAX, step=TEST_STEP)

# Detected proportions sliders
events_control = Slider(title=EVENTS_CONTROL_LABEL, value=EVENTS_CONTROL_START, start=EVENTS_CONTROL_MIN, end=EVENTS_CONTROL_MAX, step=EVENTS_CONTROL_STEP)
events_test    = Slider(title=EVENTS_TEST_LABEL,    value=EVENTS_TEST_START,    start=EVENTS_TEST_MIN,    end=EVENTS_TEST_MAX,    step=EVENTS_TEST_STEP)

ci = Slider(title=CI_LABEL, value=CI_START, start=CI_MIN, end=CI_MAX, step=CI_STEP)

# reset buttons
button  = Button(label="Reset", button_type="default")

# results section

# static labels
text_intro   = Div(text=TEXT_INTRO)
text_results = Div(text=TEXT_RESULTS)

# dynamic labels
text_risk        = Div(text='')
text_risk_ratio  = Div(text='')
text_adv_effects = Div(text='')
text_warnings    = Div(text='')

# dummy values, they are updated after update_data runs
test_risk   = 0
test_risk_l = 0
test_risk_r = 0

control_risk   = 0
control_risk_l = 0
control_risk_r = 0

# Plot


values = [ test_risk, control_risk     ]
upper  = [ test_risk_r, control_risk_r ]
lower  = [ test_risk_l, control_risk_l ]

source = ColumnDataSource(data=dict(groups=GROUPS, values=values, upper=upper, lower=lower))

p = figure(x_range=GROUPS, plot_height=350, toolbar_location=None)
p.vbar(x='groups', top='values', width=0.5, source=source, legend="groups", line_color='white')

whisker = Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay")
p.add_layout( whisker )

p.xgrid.grid_line_color = None
p.legend.visible = False

# this makes it possible to update y_range on the callback, don't remove
p.y_range=Range1d(-1, math.ceil( max (upper[0], upper[1]) ) + PLOT_RANGE_DELTA )

# update dynamic label
update_data('xxx', 0, 0)

# Assign widgets to the call back function
# updates are on value_throtled because this is too slow for realtime updates
for w in [control, test, events_control, events_test, ci]:
    w.on_change('value_throttled', update_data)

# reset button call back
button.on_click(reset_data)

left_margin   = Spacer(width=LMARGIN_WIDTH, height=400, width_policy='fixed', height_policy='auto')
middle_margin = Spacer(width=MMARGIN_WIDTH, height=400, width_policy='fixed', height_policy='auto')

# layout
inputs  = column(text_intro, test, control, events_test, events_control, ci, button)
results = column(text_results, p, text_risk, text_risk_ratio, text_adv_effects, text_warnings)

curdoc().title = PAGE_TITLE

curdoc().add_root( row(left_margin, inputs, middle_margin, results) )
