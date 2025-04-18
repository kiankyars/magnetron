# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QListWidget,
    QTextEdit, QTreeWidget, QTreeWidgetItem, QMessageBox, QSplitter
)
from magnetron.io import StorageStream

def humanize_bytes(num: int) -> str:
    """Convert bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magnetron Storage Viewer")
        self.stream: StorageStream | None = None
        self.init_ui()

    def init_ui(self):
        # Create menu bar and File menu
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = file_menu.addAction("&Open...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        # Widgets
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.array_widget = QTextEdit()
        self.array_widget.setReadOnly(True)
        self.prop_widget = QTreeWidget()
        self.prop_widget.setHeaderLabels(["Property", "Value"])

        splitter = QSplitter()
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.array_widget)
        splitter.addWidget(self.prop_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        self.setCentralWidget(splitter)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open .mag file", "", "Magnetron Files (*.mag);;All Files (*)"
        )
        if not path:
            return
        try:
            self.stream = StorageStream.open(Path(path))
            self.list_widget.clear()
            for key in self.stream.tensor_keys():
                self.list_widget.addItem(key)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_item_clicked(self, item):
        if self.stream is None:
            return
        key = item.text()
        tensor = self.stream.get(key)
        if tensor is None:
            QMessageBox.warning(self, "Not Found", f"Tensor '{key}' not found.")
            return
        # Display data
        try:
            data = tensor.tolist()
        except Exception:
            data = []
        self.array_widget.setPlainText(str(data))
        # Display properties
        self.prop_widget.clear()
        props: dict[str, any] = {
            'shape': tensor.shape,
            'rank': tensor.rank,
            'dtype': tensor.dtype,
            'numel': tensor.numel,
            'data_size': humanize_bytes(tensor.data_size),
        }
        for prop, val in props.items():
            item = QTreeWidgetItem([prop, str(val)])
            self.prop_widget.addTopLevelItem(item)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

