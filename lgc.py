''' 
Present an interactive clinical trial dimensioning interface

'''
from binoculars import binomial_confidence
from scipy import stats
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, BoxAnnotation, HoverTool, Button, Spacer, Div
from bokeh.plotting import figure

### Configuration

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
EVENTS_CONTROL_MIN   = 0
EVENTS_CONTROL_MAX   = 100
EVENTS_CONTROL_START = 5
EVENTS_CONTROL_STEP  = 0.5

EVENTS_TEST_MIN   = 0
EVENTS_TEST_MAX   = 100
EVENTS_TEST_START = 2
EVENTS_TEST_STEP  = 0.5

# Confidence level
CI_MIN   = 90
CI_MAX   = 99
CI_START = 95
CI_STEP  = 0.5

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
TEXT_RESULTS = 'Inference from experimental results:'

### End of configuration

### Functions

def mk_risk_str ( risk, risk_l, risk_r ):

    str_risk = 'Risk on control group (%) : ' + str(risk) + ' (' +  str(risk_l) + '-' + str(risk_r) + ')'

    return str_risk

# callback function for updating the data
def update_data(attrname, old, new):

    confidence_level = ci.value / 100
    z_value = stats.norm.isf( (1 - confidence_level) / 2 )

    # control group inference
    control_risk    = events_control.value
    control_risk_ci = np.array( binomial_confidence(control_risk / 100, control.value, z=z_value) ) * 100
    control_risk_l  = round(control_risk_ci[0], 2)
    control_risk_r  = round(control_risk_ci[1], 2)

    str_control_risk = mk_risk_str (control_risk, control_risk_l, control_risk_r)

    # test group inference
    test_risk    = events_test.value
    test_risk_ci = np.array( binomial_confidence(test_risk / 100, test.value, z=z_value) ) * 100
    test_risk_l  = round(test_risk_ci[0], 2)
    test_risk_r  = round(test_risk_ci[1], 2)

    str_test_risk = mk_risk_str (test_risk, test_risk_l, test_risk_r)

    # risk ratio
    str_risk_ratio = 'Risk ratio: ' + str ( round ( test_risk / control_risk, 2 ) )

    # adverse effects threshold
    # to find at least one case at the current confidence level the probability must be this or higher
    adv_effects_threshold = 1 - ( 1 - confidence_level )**( 1 / test.value ) 
    str_adv_effects = 'Adverse effects detectability threshold (%): ' + str( round (adv_effects_threshold * 100, 2) )

    text_risk.text        = str_control_risk + '<br/>' + str_test_risk
    text_risk_ratio.text  = str_risk_ratio
    text_adv_effects.text = str_adv_effects

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

# update dynamic labels
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
inputs  = column(text_intro, control, test, events_control, events_test, ci, button)
results = column(text_results, text_risk, text_risk_ratio, text_adv_effects)

curdoc().title = PAGE_TITLE

curdoc().add_root( row(left_margin, inputs, middle_margin, results) )
