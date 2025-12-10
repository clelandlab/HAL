import yaml
import ipywidgets as widgets
from IPython.display import display, Markdown
from IPython.core.getipython import get_ipython

log_output = widgets.Output(layout={'height': '300px', 'overflow_y': 'auto'})
log_accordion = widgets.Accordion(children=[log_output], titles=["[HAL]"])
sequence_accordion = widgets.Accordion()

def get_markdown_output(content):
    out = widgets.Output()
    with out:
        display(Markdown(content))
    return out

def init():
    display(log_accordion)
    display(sequence_accordion)

def log(content, status="Idle"):
    log_accordion.set_title(0, f"Status [HAL] {status}")
    with log_output:
        print(content)

show = lambda x: display(Markdown("---\n\n" + x + "\n\n---\n\n"))

def new_cell(content):
    get_ipython().set_next_input(content, replace=False)

def sequence(seq):
    sequence_accordion.children = []
    for i, step in enumerate(seq):
        tab = widgets.Tab()
        for key in step:
            if key == "_type":
                continue
            content = step[key]
            if "code" in key:
                content = f"```python\n{content}\n```"
            if key == "_doc":
                content = f"```yaml\n{yaml.dump(content)}\n```"
            tab.children += (get_markdown_output(content),)
            tab.set_title(len(tab.children) - 1, key)
        tab.selected_index = 1 if len(tab.children) > 1 else 0
        sequence_accordion.children += (tab,)
        sequence_accordion.set_title(i, f"sequence [{i}] {step.get('_type', '')}")

def docs(doc_texts):
    n = len(doc_texts)
    out = widgets.Output()
    def on_slider_change(change):
        i = change['new']
        out.clear_output()
        with out:
            display(Markdown(doc_texts[i]))
    slider = widgets.IntSlider(value=0, min=0, max=n-1, step=1, description=f'Docs ({n}):', continuous_update=False)
    slider.observe(on_slider_change, names='value')
    on_slider_change({'new': 0})
    display(slider, out)
