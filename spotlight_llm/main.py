import sys
import os
import json
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QVBoxLayout, QTextEdit, QDesktopWidget, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QPainter, QColor, QFont
from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
import threading

class StreamHandler(BaseCallbackHandler):
    def __init__(self, text_widget, response_callback):
        self.text_widget = text_widget
        self.response_callback = response_callback
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text_widget.insertPlainText(token)
        self.text_widget.ensureCursorVisible()
        self.full_response += token

    def on_llm_end(self, response, **kwargs):
        self.response_callback(self.full_response)

class SpotlightLLM(QWidget):
    def __init__(self):
        super().__init__()
        self.models = ["qwen2:0.5b", "qwen2:0.5b-instruct", "gemma2:2b"]
        self.current_model = self.models[0]
        self.initUI()
        self.setup_data_directory()
        self.session_interactions = []  # New list to store all interactions

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Create a horizontal layout for search bar and model selection
        search_layout = QHBoxLayout()

        self.search_bar = QLineEdit(self)
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 20px;
                padding: 10px;
                color: white;
                font-size: 18px;
            }
        """)
        self.search_bar.returnPressed.connect(self.on_submit)
        search_layout.addWidget(self.search_bar, 7)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.models)
        self.model_selector.setStyleSheet("""
            QComboBox {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 20px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                selection-background-color: rgba(80, 80, 80, 200);
                color: white;
            }
        """)
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        search_layout.addWidget(self.model_selector, 3)

        layout.addLayout(search_layout)

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 10px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
        """)
        self.result_area.hide()
        layout.addWidget(self.result_area)

        self.setLayout(layout)

        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        screen_size = QDesktopWidget().screenGeometry(screen)
        window_width = 750
        window_height = 60
        x = (screen_size.width() - window_width) // 2
        y = screen_size.height() // 4
        self.setGeometry(x, y, window_width, window_height)

    def setup_data_directory(self):
        self.data_dir = os.path.join(os.path.expanduser("~"), "spotlight_llm_data")
        os.makedirs(self.data_dir, exist_ok=True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(30, 30, 30, 200))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

    def on_model_change(self, model):
        self.current_model = model

    def on_submit(self):
        query = self.search_bar.text()
        if query:
            self.result_area.clear()
            self.result_area.show()
            self.animate_expand()
            self.current_query = query
            self.current_timestamp = datetime.now().isoformat()
            threading.Thread(target=self.get_ollama_response, args=(query,), daemon=True).start()

    def get_ollama_response(self, prompt):
        stream_handler = StreamHandler(self.result_area, self.save_interaction)
        llm = Ollama(model=self.current_model, callbacks=[stream_handler])
        llm(prompt)

    def save_interaction(self, response):
        interaction_data = {
            "timestamp": self.current_timestamp,
            "model": self.current_model,
            "query": self.current_query,
            "response": response
        }
        self.session_interactions.append(interaction_data)  # Add to session interactions

    def animate_expand(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(self.geometry())
        new_height = 400
        new_geometry = QRect(self.x(), self.y(), self.width(), new_height)
        self.animation.setEndValue(new_geometry)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.save_session()  # Save the entire session before closing
            self.close()

    def save_session(self):
        if self.session_interactions:
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(self.session_interactions, f, indent=2)

    def closeEvent(self, event):
        self.save_session()  # Save the session when the window is closed
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpotlightLLM()
    ex.show()
    sys.exit(app.exec_())
