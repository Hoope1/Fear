# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from processing.batch_processor import BatchProcessor


class EdgeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Modern Edge Detection")
        self.setGeometry(100, 100, 600, 400)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Folder selection
        folder_group = QGroupBox("Input Folder")
        folder_layout = QHBoxLayout()

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 5px; }"
        )
        folder_layout.addWidget(self.folder_label)

        self.select_button = QPushButton("Select Folder")
        self.select_button.clicked.connect(self.select_folder)
        self.select_button.setMinimumWidth(120)
        folder_layout.addWidget(self.select_button)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()

        # Overall progress
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall:"))
        self.overall_progress = QProgressBar()
        overall_layout.addWidget(self.overall_progress)
        progress_layout.addLayout(overall_layout)

        # Current file
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Current:"))
        self.current_label = QLabel("Waiting...")
        self.current_label.setMinimumWidth(200)
        current_layout.addWidget(self.current_label)
        current_layout.addStretch()
        progress_layout.addLayout(current_layout)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Status bar
        self.status_label = QLabel("Ready to process images")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "QLabel { background-color: #e0e0e0; padding: 5px; }"
        )
        layout.addWidget(self.status_label)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_label.setText(folder)
            self.log_text.append(f"Selected folder: {folder}")
            self.process_folder(folder)

    def process_folder(self, folder_path):
        # Disable button during processing
        self.select_button.setEnabled(False)
        self.status_label.setText("Processing...")

        # Create and start processor thread
        self.processor_thread = BatchProcessor(folder_path)
        self.processor_thread.progress_update.connect(self.update_progress)
        self.processor_thread.log_message.connect(self.log_text.append)
        self.processor_thread.processing_complete.connect(self.processing_finished)
        self.processor_thread.current_file_update.connect(self.update_current_file)
        self.processor_thread.start()

    def update_progress(self, value):
        self.overall_progress.setValue(value)

    def update_current_file(self, filename):
        self.current_label.setText(filename)

    def processing_finished(self):
        self.select_button.setEnabled(True)
        self.status_label.setText("Processing complete!")
        self.current_label.setText("Done")
        self.log_text.append("\n✓ All images processed successfully!")
        self.log_text.append(f"✓ Results saved in: {os.path.abspath('output/')}")

