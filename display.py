import ipywidgets as widgets
from IPython.display import display, Markdown

log_output = widgets.Output(layout={'height': '200px', 'overflow_y': 'auto'})

def init():
    accordion = widgets.Accordion(children=[log_output], titles=["HAL Log"])
    accordion.selected_index = 0
    display(accordion)

def log(content):
    with log_output:
        print(content)

show = lambda x: display(Markdown("---\n\n" + x + "\n\n---\n\n"))
