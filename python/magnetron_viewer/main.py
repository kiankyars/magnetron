# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import time
import sys
import os
from magnetron.core import Tensor
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

FONT_SIZE: int = 14


def process_events_idle():
    for _ in range(0, 5):  # Sleep for a bit to allow the loading box to show up
        QApplication.processEvents()
        time.sleep(0.1)


class MAGNETRONViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window_icon = QIcon('logo.png')
        self.tensor_icon = QIcon('icons/_ptr.png')
        self.folder_icon = QIcon('icons/folder.png')
        self.metadata_icon = QIcon('icons/metadata.png')
        self.setWindowTitle('magnetron File Viewer')
        self.setWindowIcon(self.window_icon)
        self.resize(1920, 1080)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)

        self.tensor_tree = QTreeWidget()
        self.tensor_tree.setHeaderHidden(True)
        self.tensor_tree.setStyleSheet(f'font-size: {FONT_SIZE}px;')
        self.tensor_tree.itemClicked.connect(self.show_tensor_data)

        self.data_view_container = QWidget()
        data_view_layout = QVBoxLayout(self.data_view_container)

        self.data_view = QPlainTextEdit()
        self.data_view.setReadOnly(True)
        self.data_view.setStyleSheet(f'font-size: {FONT_SIZE}px;')

        data_view_layout.addWidget(self.data_view)
        data_view_layout.setContentsMargins(0, 0, 0, 0)

        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setStyleSheet(f'font-size: {FONT_SIZE}px;')

        splitter.addWidget(self.tensor_tree)
        splitter.addWidget(self.data_view_container)
        splitter.addWidget(self.info_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 8)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        self.tensors = {}
        self.metadata = {}

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open magnetron File', os.getcwd(),
                                                   'magnetron Files (*.magnetron);;All Files (*)')
        if not file_name:
            print('No file selected')
            return

        self.setWindowTitle(f'magnetron File Viewer - {os.path.basename(file_name)}')

        self.tensor_tree.clear()
        self.tensors.clear()
        self.metadata.clear()

        tensor_file = Tensor.load(file_name)
        tensors_item = QTreeWidgetItem(self.tensor_tree)
        tensors_item.setText(0, 'Tensors')
        tensors_item.setIcon(0, self.folder_icon)
        for tensor in [tensor_file]:
            tensor.name = tensor.name or f'Tensor {len(self.tensors) + 1}'
            self.tensors[tensor.name] = tensor
            tensor_item = QTreeWidgetItem(tensors_item)
            tensor_item.setText(0, tensor.name)
            tensor_item.setIcon(0, self.tensor_icon)

        metadata_item = QTreeWidgetItem(self.tensor_tree)
        metadata_item.setText(0, 'Metadata')
        metadata_item.setIcon(0, self.folder_icon)
        metadata = {
            'File Name': os.path.basename(file_name),
            'File Size': os.path.getsize(file_name),
            'File Path': file_name
        }
        for key, value in metadata.items():
            self.metadata[key] = value
            metadata_entry = QTreeWidgetItem(metadata_item)
            metadata_entry.setText(0, f'{key}: {value}')
            metadata_entry.setIcon(0, self.metadata_icon)

        self.tensor_tree.expandAll()

    def show_tensor_data(self, item):
        parent = item.parent()
        if parent is None or parent.text(0) != 'Tensors':
            return

        tensor_name = item.text(0)
        if tensor_name not in self.tensors:
            return
        tensor = self.tensors[tensor_name]
        tensor_data = tensor.tolist()

        rows = []
        elements_per_row = 16
        for i in range(0, len(tensor_data), elements_per_row):
            row = '  '.join(f'{value:10.5f}' for value in tensor_data[i:i + elements_per_row])
            rows.append(row)
        data_str = '\n'.join(rows)
        self.data_view.setPlainText(data_str)

        extra_info = [
            f'Name: {tensor.name}',
            f'Dimensions: {tensor.shape}',
            f'Strides: {tensor.strides}',
            f'DType: {tensor.dtype}',
            f'Rank: {tensor.rank}',
            f'Total Elements: {tensor.numel}',
            f'Total Bytes: {tensor.data_size}',
            f'Min: {min(tensor_data)}',
            f'Max: {max(tensor_data)}',
            f'Mean: {sum(tensor_data) / len(tensor_data)}',
            f'Transposed: {tensor.is_transposed}',
            f'Permuted: {tensor.is_permuted}',
            f'Contiguous: {tensor.is_contiguous}',
        ]
        self.info_panel.setText('\n'.join(extra_info))


def main():
    app = QApplication(sys.argv)
    viewer = MAGNETRONViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
