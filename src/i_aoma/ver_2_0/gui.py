import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel


def main():
    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("Responsive Window")

    layout = QVBoxLayout()

    label = QLabel("This is a label")
    label.setAlignment(Qt.AlignCenter)  # Center the text within the label

    button1 = QPushButton("Button 1")
    button2 = QPushButton("Button 2")

    layout.addWidget(label)
    layout.addWidget(button1)
    layout.addWidget(button2)

    window.setLayout(layout)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
