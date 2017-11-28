from IPython.display import display
import ipywidgets as widgets
import functools


class AnnotationPane(object):
    LABELS = [
        ('Yes', True),
        ('No', False),
    ]

    def __init__(self, dataset, sampler):
        self.text_pane = widgets.HTML()
        self.buttons = []
        for desc, label in self.LABELS:
            button = widgets.Button(
                description=desc
            )
            button.on_click(functools.partial(self.on_click, label))
            self.buttons.append(button)
        self.view = widgets.VBox([
            widgets.HBox(self.buttons),
            self.text_pane,
        ])
        self.queue = [text for text, label in sampler(dataset)]
        self.dataset = dataset
        self.draw()
        display(self.view)

    def on_click(self, label, button):
        self.dataset.add_label(self.text, label)
        self.draw()

    def draw(self):
        if not self.queue:
            for button in self.buttons:
                button.disabled = True
            self.text_pane.value = '<p>Finished</p>'
        else:
            self.text = self.queue.pop(0)
            self.text_pane.value = '<p>{}</p>'.format(self.text)
